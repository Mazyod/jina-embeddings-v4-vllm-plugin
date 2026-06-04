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
vllm_image = _with_local(
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm", "pillow", "numpy>=2.0", "huggingface_hub")
    .env({"HF_HOME": CACHE})
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
