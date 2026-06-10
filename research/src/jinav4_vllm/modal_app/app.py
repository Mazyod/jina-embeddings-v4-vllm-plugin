"""Modal app: images, volumes, secret, and the artifact-building jobs (extract + bake).

The serving entrypoint is `serve_c.py`; the validation loop is `reference.py` + `offline.py` +
`collect.py` + `jinav4_vllm.eval.report`; post-upgrade API checks live in `revalidate.py`.
"""
from __future__ import annotations
import modal

app = modal.App("jinav4-vllm")

# Persistent volumes for HF cache and our artifacts.
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)
artifacts = modal.Volume.from_name("jinav4-artifacts", create_if_missing=True)
CACHE = "/root/.cache/huggingface"
VLLM_CACHE = "/root/.cache/vllm"
ART = "/artifacts"

# Chat template ships with the root plugin package (one level up from research/); _with_local mounts
# it at this stable path so the bake job can read it without installing the plugin.
CHAT_TEMPLATE_SRC = "/root/jina_image_chat_template.jinja"


def _with_local(img):
    """Attach project source + probe images so containers can import jinav4_vllm and read probes.

    The image chat template ships with the root plugin package (one level up from research/); mount
    it at a stable path so the bake job can read it without installing the plugin.
    """
    return (
        img.add_local_dir("src/jinav4_vllm", remote_path="/root/jinav4_vllm")
           .add_local_dir("data/probes", remote_path="/root/data/probes")
           .add_local_file(
               "../src/jina_v4_vllm_plugin/jina_image_chat_template.jinja",
               remote_path="/root/jina_image_chat_template.jinja",
           )
    )


# Reference image: transformers stack.
ref_image = _with_local(
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch", "torchvision", "transformers>=4.52,<5", "peft>=0.11", "safetensors",
        "huggingface_hub", "pillow", "numpy>=2.0", "accelerate",
    )
    .env({"HF_HOME": CACHE})
)

# vLLM image: pin at lock time (see docs/COMPAT.md). Kept separate from the transformers stack.
_vllm_base = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm", "pillow", "numpy>=2.0", "huggingface_hub")
    .env({"HF_HOME": CACHE})
)
vllm_image = _with_local(_vllm_base)

# Serving image: install our out-of-tree model as a vLLM general plugin (entry point), so a stock
# `vllm serve ... --hf-overrides architectures=[JinaV4MultiVector]` emits final [n,128].
# copy=True lets the pip-install build step run before the (non-copy) runtime mounts in _with_local.
# Install our out-of-tree model as a vLLM general plugin (entry point). The plugin is now the repo's
# ROOT package (one level up from research/); assemble its project layout under /opt/jina_plugin and
# pip-install it. Post-publish you can replace the three add_local_* + install with a single:
#   .run_commands("python -m pip install --no-deps jina-v4-vllm-plugin")
vllm_plugin_image = _with_local(
    _vllm_base
    .add_local_file("../pyproject.toml", remote_path="/opt/jina_plugin/pyproject.toml", copy=True)
    .add_local_file("../README.md", remote_path="/opt/jina_plugin/README.md", copy=True)
    .add_local_dir("../src/jina_v4_vllm_plugin",
                   remote_path="/opt/jina_plugin/src/jina_v4_vllm_plugin", copy=True)
    .run_commands(
        "python -m pip install --no-deps /opt/jina_plugin "
        "|| (python -m ensurepip && python -m pip install --no-deps /opt/jina_plugin)"
    )
)

HF_SECRET = modal.Secret.from_name("huggingface-secret")   # holds HF_TOKEN
GPU = "A10G"
COMMON = dict(volumes={CACHE: hf_cache, VLLM_CACHE: vllm_cache, ART: artifacts}, secrets=[HF_SECRET])


@app.function(image=ref_image, timeout=1800, **COMMON)
def extract_projector():
    import sys; sys.path.insert(0, "/root")
    from jinav4_vllm.projector.extract import extract_and_save
    meta = extract_and_save(f"{ART}/projector/retrieval.npz")
    artifacts.commit()
    print(meta)
    return meta


@app.function(image=ref_image, timeout=3600, **COMMON)
def bake_checkpoint(out_dir: str = f"{ART}/jina-v4-mv-baked",
                    src_model: str = "jinaai/jina-embeddings-v4-vllm-retrieval",
                    min_pixels: int = 0, max_pixels: int = 0):
    """Produce a fully self-contained, drop-in checkpoint for the native server.

    = the vLLM retrieval checkpoint + the multi_vector_projector tensors + architectures override
      + the Jina image chat template — so `vllm serve <out_dir> --runner pooling
      --pooler-config.task token_embed` (with the plugin installed) works with NO --hf-overrides,
      NO --chat-template, and NO projector env var.
    """
    import os, json, shutil
    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from huggingface_hub import snapshot_download

    repo = snapshot_download(src_model)
    os.makedirs(out_dir, exist_ok=True)

    # 1) copy every checkpoint file (resolve symlinks) into out_dir
    for fn in os.listdir(repo):
        src = os.path.join(repo, fn)
        if os.path.isfile(src):
            shutil.copy2(os.path.realpath(src), os.path.join(out_dir, fn))

    # 2) determine the existing weight->shard map (sharded index or a single safetensors)
    index_path = os.path.join(out_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        index = json.load(open(index_path))
        weight_map = index["weight_map"]
        metadata = index.get("metadata", {})
    else:
        # single-file checkpoint: build an index that maps all of its tensors to model.safetensors
        weight_map, metadata = {}, {}
        single = os.path.join(out_dir, "model.safetensors")
        with safe_open(single, framework="pt") as f:
            for k in f.keys():
                weight_map[k] = "model.safetensors"

    # 3) write the projector tensors as a new shard
    proj = np.load(f"{ART}/projector/retrieval.npz")
    proj_tensors = {
        "multi_vector_projector.weight": torch.from_numpy(proj["W"]).to(torch.float32),
        "multi_vector_projector.bias": torch.from_numpy(proj["b"]).to(torch.float32),
    }
    save_file(proj_tensors, os.path.join(out_dir, "model-projector.safetensors"))
    weight_map["multi_vector_projector.weight"] = "model-projector.safetensors"
    weight_map["multi_vector_projector.bias"] = "model-projector.safetensors"
    metadata["total_size"] = int(metadata.get("total_size", 0)) + int(
        proj_tensors["multi_vector_projector.weight"].numel() * 4
        + proj_tensors["multi_vector_projector.bias"].numel() * 4)
    json.dump({"metadata": metadata, "weight_map": weight_map}, open(index_path, "w"), indent=2)

    # 4) architectures -> JinaV4MultiVector (resolved by the installed plugin)
    cfg_path = os.path.join(out_dir, "config.json")
    cfg = json.load(open(cfg_path))
    cfg["architectures"] = ["JinaV4MultiVector"]
    json.dump(cfg, open(cfg_path, "w"), indent=2)

    # 4b) image fidelity: bake min/max pixels into the image processor config (drop-in)
    if min_pixels or max_pixels:
        pp_path = os.path.join(out_dir, "preprocessor_config.json")
        pp = json.load(open(pp_path)) if os.path.exists(pp_path) else {}
        if min_pixels:
            pp["min_pixels"] = int(min_pixels)
        if max_pixels:
            pp["max_pixels"] = int(max_pixels)
        # newer Qwen2.5-VL processors also read size={shortest_edge,longest_edge}
        size = pp.get("size", {}) if isinstance(pp.get("size"), dict) else {}
        if min_pixels:
            size["shortest_edge"] = int(min_pixels)
        if max_pixels:
            size["longest_edge"] = int(max_pixels)
        if size:
            pp["size"] = size
        json.dump(pp, open(pp_path, "w"), indent=2)

    # 5) bake the Jina image chat template so multimodal /pooling needs no --chat-template
    tmpl = open(CHAT_TEMPLATE_SRC).read()
    open(os.path.join(out_dir, "chat_template.jinja"), "w").write(tmpl)
    # The Qwen2.5-VL *processor* template (chat_template.json) wins for multimodal; overwrite it,
    # plus tokenizer_config for completeness.
    json.dump({"chat_template": tmpl}, open(os.path.join(out_dir, "chat_template.json"), "w"), indent=2)
    tok_cfg_path = os.path.join(out_dir, "tokenizer_config.json")
    if os.path.exists(tok_cfg_path):
        tok_cfg = json.load(open(tok_cfg_path))
        tok_cfg["chat_template"] = tmpl
        json.dump(tok_cfg, open(tok_cfg_path, "w"), indent=2)

    artifacts.commit()
    files = sorted(os.listdir(out_dir))
    meta = {"out_dir": out_dir, "architectures": cfg["architectures"],
            "has_index": True, "n_files": len(files),
            "projector_in_map": "multi_vector_projector.weight" in weight_map}
    print(meta)
    return meta
