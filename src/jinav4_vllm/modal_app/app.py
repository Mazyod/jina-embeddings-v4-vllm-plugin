"""Modal app: images, volume, secret, and the foundational GPU jobs."""
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


def _with_local(img):
    """Attach project source + probe images so containers can import jinav4_vllm and read probes."""
    return (
        img.add_local_dir("src/jinav4_vllm", remote_path="/root/jinav4_vllm")
           .add_local_dir("data/probes", remote_path="/root/data/probes")
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

# vLLM image: pin at lock time (see spike_r3). Kept separate from transformers stack.
_vllm_base = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm", "pillow", "numpy>=2.0", "huggingface_hub")
    .env({"HF_HOME": CACHE})
)
vllm_image = _with_local(_vllm_base)

# Variant C image: install our out-of-tree model as a vLLM general plugin (entry point), so a
# stock `vllm serve ... --hf-overrides architectures=[JinaV4MultiVector]` emits final [n,128].
# copy=True lets the pip-install build step run before the (non-copy) runtime mounts in _with_local.
vllm_plugin_image = _with_local(
    _vllm_base
    .add_local_dir("src/jinav4_vllm/vllm_plugin", remote_path="/opt/jina_plugin", copy=True)
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


@app.function(image=ref_image, gpu=GPU, timeout=1800, **COMMON)
def verify_projector():
    """R4: our extracted W,b must reproduce the model's own retrieval projector."""
    import sys; sys.path.insert(0, "/root")
    import numpy as np, torch
    from transformers import AutoModel
    proj = np.load(f"{ART}/projector/retrieval.npz")
    W, b = proj["W"], proj["b"]
    model = AutoModel.from_pretrained("jinaai/jina-embeddings-v4", trust_remote_code=True,
                                      torch_dtype=torch.float32).eval()
    # Activate retrieval adapter if the model exposes adapter switching.
    for setter in ("set_adapter", "set_task"):
        if hasattr(model, setter):
            try: getattr(model, setter)("retrieval")
            except Exception: pass
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.standard_normal((16, 2048)), dtype=torch.float32)
    with torch.no_grad():
        ref = model.multi_vector_projector(x).cpu().numpy()
    ours = x.numpy() @ W.T + b
    mad = float(np.abs(ref - ours).max())
    print(f"max_abs_diff={mad:.6e}", "PASS" if mad < 1e-3 else "FAIL")
    assert mad < 1e-3, "extracted projector does not match model projector"
    return mad


@app.function(image=ref_image, timeout=3600, **COMMON)
def bake_checkpoint(out_dir: str = f"{ART}/jina-v4-mv-baked",
                    src_model: str = "jinaai/jina-embeddings-v4-vllm-retrieval",
                    min_pixels: int = 0, max_pixels: int = 0):
    """Produce a fully self-contained, drop-in checkpoint for Variant C.

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
    tmpl = open("/root/jinav4_vllm/vllm_plugin/jina_image_chat_template.jinja").read()
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


@app.function(image=vllm_image, timeout=1800, **COMMON)
def recon_vllm_pooling():
    """CPU recon: learn the exact vLLM 0.22 API to add an in-engine projected multi-vector pooler.

    Dumps: registered architectures (Col*/embed/Qwen-VL), the pooling-adapter source
    (how --convert embed attaches a pooler), the pooler module surface, and an existing
    projected-multivector model (ColQwen/ColPali) to copy the pattern.
    """
    import inspect, os, pkgutil, importlib
    out = []
    def dump(title, s, cap=6000):
        out.append(f"\n===== {title} =====\n{s[:cap]}")

    # 1) registered architectures of interest
    try:
        from vllm import ModelRegistry
        archs = sorted(ModelRegistry.get_supported_archs())
        interesting = [a for a in archs if any(k in a.lower() for k in
                       ("col", "embed", "qwen2_5_vl", "qwen2vl", "jina", "gme", "llavanext", "vlm2vec"))]
        dump("ARCHS (interesting)", "\n".join(interesting))
    except Exception as e:
        dump("ARCHS error", repr(e))

    # 2) the embedding/pooling adapter (how generative -> pooling)
    for mod in ("vllm.model_executor.models.adapters",):
        try:
            m = importlib.import_module(mod)
            names = [n for n in dir(m) if not n.startswith("_")]
            dump(f"{mod} names", ", ".join(names))
            for fn in ("as_embedding_model", "as_seq_cls_model", "_create_pooling_model_cls", "as_reward_model"):
                if hasattr(m, fn):
                    try: dump(f"{mod}.{fn}", inspect.getsource(getattr(m, fn)))
                    except Exception as e: dump(f"{mod}.{fn} src err", repr(e))
        except Exception as e:
            dump(f"{mod} err", repr(e))

    # 3) pooler module surface + base classes
    try:
        import vllm.model_executor.layers.pooler as P
        dump("pooler public names", ", ".join(n for n in dir(P) if not n.startswith("_")))
        for cls in ("Pooler", "DispatchPooler", "PoolerHead", "PoolingType", "PoolerOutput",
                    "PoolingParamsUpdate", "PoolingMetadata", "build_output", "AllPool", "EmbeddingPoolerHead"):
            obj = getattr(P, cls, None)
            if obj is None: continue
            try:
                if inspect.isclass(obj):
                    meths = [n for n in dir(obj) if not n.startswith("_")]
                    sig = ""
                    for mm in ("__init__", "forward", "for_encode", "for_embed", "get_pooling_updates"):
                        f = getattr(obj, mm, None)
                        if f:
                            try: sig += f"\n  {mm}{inspect.signature(f)}"
                            except Exception: pass
                    dump(f"pooler.{cls}", f"methods={meths}\nsigs={sig}")
                else:
                    dump(f"pooler.{cls}", repr(obj))
            except Exception as e:
                dump(f"pooler.{cls} err", repr(e))
    except Exception as e:
        dump("pooler import err", repr(e))

    # 4) find an existing projected-multivector model to copy (colqwen/colpali)
    try:
        import vllm.model_executor.models as M
        pkgdir = os.path.dirname(M.__file__)
        cand = [f for f in os.listdir(pkgdir) if any(k in f.lower() for k in ("col", "jina", "gme"))]
        dump("model files (col/jina/gme)", ", ".join(sorted(cand)))
        for f in cand:
            if f.endswith(".py"):
                src = open(os.path.join(pkgdir, f)).read()
                # show the pooler-related parts
                lines = src.splitlines()
                hits = [i for i,l in enumerate(lines) if any(k in l for k in
                        ("pooler", "Pooler", "token_embed", "projector", "projection", "multi_vector", "nn.Linear", "normalize"))]
                snippet = "\n".join(f"{i:4d}: {lines[i]}" for i in hits[:60])
                dump(f"model {f} (pooler/projection lines)", snippet, cap=4000)
    except Exception as e:
        dump("models scan err", repr(e))

    text = "".join(out)
    print(text)
    return {"len": len(text)}


@app.function(image=vllm_image, timeout=1800, **COMMON)
def recon_dump_sources():
    """Dump full source of the closest template model + the token-embed pooler builder."""
    import os, importlib
    import vllm.model_executor.models as M
    pkg = os.path.dirname(M.__file__)
    out = []
    def cat(path, title, cap=20000):
        try:
            out.append(f"\n########## {title} ({path}) ##########\n" + open(path).read()[:cap])
        except Exception as e:
            out.append(f"\n########## {title} ERR {e!r} ##########")
    cat(os.path.join(pkg, "colqwen3.py"), "colqwen3.py")
    # the token-embed pooler builder + TokenPooler
    import vllm.model_executor.layers.pooler.tokwise as TW
    cat(TW.__file__, "pooler/tokwise.py")
    # colpali load_weights projector pattern (smaller, for reference)
    cat(os.path.join(pkg, "colpali.py"), "colpali.py", cap=14000)
    text = "".join(out)
    print(text)
    return {"len": len(text)}


@app.function(image=vllm_image, timeout=1800, **COMMON)
def recon_qwen25_api():
    """Confirm Qwen2.5-VL class names + import paths for the JinaV4MultiVector plugin."""
    import inspect, importlib
    out = {}
    m = importlib.import_module("vllm.model_executor.models.qwen2_5_vl")
    out["qwen2_5_vl names"] = [n for n in dir(m) if "Qwen2_5_VL" in n]
    for path, names in {
        "vllm.model_executor.layers.pooler.tokwise": ["pooler_for_token_embed"],
        "vllm.model_executor.models.interfaces": ["SupportsLateInteraction", "SupportsMultiModal"],
        "vllm.model_executor.models.interfaces_base": ["default_pooling_type"],
        "vllm.model_executor.models.utils": ["AutoWeightsLoader", "WeightsMapper"],
        "vllm.model_executor.model_loader.weight_utils": ["default_weight_loader"],
        "vllm.multimodal": ["MULTIMODAL_REGISTRY"],
    }.items():
        try:
            mod = importlib.import_module(path)
            out[path] = {n: hasattr(mod, n) for n in names}
        except Exception as e:
            out[path] = f"ERR {e!r}"
    try:
        from vllm.model_executor.layers.pooler.tokwise import pooler_for_token_embed
        out["pooler_for_token_embed sig"] = str(inspect.signature(pooler_for_token_embed))
    except Exception as e:
        out["pooler_for_token_embed sig"] = f"ERR {e!r}"
    # head_dtype presence on a built model_config? just check attribute name exists on class
    try:
        from vllm.config import ModelConfig
        out["ModelConfig has head_dtype"] = "head_dtype" in dir(ModelConfig)
    except Exception as e:
        out["ModelConfig head_dtype"] = f"ERR {e!r}"
    import json; print(json.dumps(out, indent=2, default=str))
    return out


@app.function(image=vllm_image, gpu=GPU, timeout=1800, **COMMON)
def inspect_pooler():
    """R5 recon: learn vLLM 0.22's model class + pooler interface for an in-engine projection."""
    import inspect
    from vllm import LLM
    from vllm.config import PoolerConfig
    llm = LLM(model="jinaai/jina-embeddings-v4-vllm-retrieval", runner="pooling",
              pooler_config=PoolerConfig(task="token_embed"), max_model_len=512,
              gpu_memory_utilization=0.5, enforce_eager=True)
    info = {}
    try:
        runner = llm.llm_engine.model_executor.driver_worker.model_runner
        model = runner.model
    except Exception as e:
        # vLLM 0.22 v1 engine path differs; try the v1 core
        info["driver_path_error"] = repr(e)
        model = None
        for attr in ("llm_engine", "engine"):
            eng = getattr(llm, attr, None)
            if eng is not None:
                info["engine_attrs"] = [a for a in dir(eng) if not a.startswith("__")][:40]
                break
    if model is not None:
        info["model_class"] = type(model).__module__ + "." + type(model).__qualname__
        info["model_mro"] = [c.__module__ + "." + c.__name__ for c in type(model).__mro__]
        info["has_pooler"] = hasattr(model, "pooler")
        if hasattr(model, "pooler"):
            p = model.pooler
            info["pooler_class"] = type(p).__module__ + "." + type(p).__qualname__
            info["pooler_callables"] = [a for a in dir(p) if not a.startswith("_")][:40]
            try:
                info["pooler_forward_sig"] = str(inspect.signature(p.forward))
            except Exception as e:
                info["pooler_forward_sig_err"] = repr(e)
    print(info)
    return info


@app.function(image=vllm_image, gpu=GPU, timeout=1800, **COMMON)
def spike_r3_offline_shape():
    """R3: confirm offline pooling returns [n, 2048] per-token hidden states + token ids."""
    from vllm import LLM
    from vllm.config import PoolerConfig
    llm = LLM(model="jinaai/jina-embeddings-v4-vllm-retrieval",
              runner="pooling", pooler_config=PoolerConfig(task="token_embed"),
              max_model_len=1024, gpu_memory_utilization=0.8)
    out = llm.encode(["Query: hello world"])[0]
    data = out.outputs.data
    shape = tuple(data.shape)
    ntok = len(out.prompt_token_ids)
    print(f"data.shape={shape} n_prompt_tokens={ntok}")
    assert shape[1] == 2048, f"expected hidden dim 2048, got {shape[1]}"
    assert shape[0] == ntok, "row count must equal prompt token count"
    return {"shape": shape, "n_tokens": ntok}
