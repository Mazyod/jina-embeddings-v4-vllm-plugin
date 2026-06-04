"""Extract the retrieval-effective multi_vector_projector and save to the volume.

Run: uv run modal run src/jinav4_vllm/modal_app/app.py::extract_projector
Produces /artifacts/projector/retrieval.npz on the Modal volume: {W:[128,2048], b:[128]}.
"""
from __future__ import annotations


def extract_and_save(out_path: str) -> dict:
    """Executed inside the Modal ref_image container. Returns metadata dict."""
    import json, os
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from peft import PeftConfig
    from jinav4_vllm.projector.merge import merge_linear

    MAIN = "jinaai/jina-embeddings-v4"
    repo = snapshot_download(MAIN)

    # --- locate base projector weights across shards ---
    base_W = base_b = None
    for fn in os.listdir(repo):
        if fn.endswith(".safetensors"):
            with safe_open(os.path.join(repo, fn), framework="pt") as f:
                for k in f.keys():
                    if k.endswith("multi_vector_projector.weight"):
                        base_W = f.get_tensor(k).float().cpu().numpy()
                    elif k.endswith("multi_vector_projector.bias"):
                        base_b = f.get_tensor(k).float().cpu().numpy()
    assert base_W is not None and base_b is not None, "base projector not found"
    assert base_W.shape == (128, 2048), f"unexpected base_W shape {base_W.shape}"

    # --- locate retrieval adapter LoRA for the projector ---
    adapter_dir = os.path.join(repo, "adapters", "retrieval")
    if not os.path.isdir(adapter_dir):
        # fall back: adapters may be under a flat 'adapters' dir or named differently
        for cand in ("adapters/retrieval", "retrieval", "adapters"):
            p = os.path.join(repo, cand)
            if os.path.isdir(p) and any(x.startswith("adapter_") for x in os.listdir(p)):
                adapter_dir = p
                break
    cfg = PeftConfig.from_pretrained(adapter_dir)
    alpha, r = cfg.lora_alpha, cfg.r

    lora_A = lora_B = None
    for fn in os.listdir(adapter_dir):
        if fn.endswith(".safetensors"):
            with safe_open(os.path.join(adapter_dir, fn), framework="pt") as f:
                for k in f.keys():
                    if "multi_vector_projector" in k and "lora_A" in k:
                        lora_A = f.get_tensor(k).float().cpu().numpy()
                    elif "multi_vector_projector" in k and "lora_B" in k:
                        lora_B = f.get_tensor(k).float().cpu().numpy()

    if lora_A is not None and lora_B is not None:
        W, b = merge_linear(base_W, base_b, lora_A, lora_B, alpha=alpha, r=r)
        merged = True
    else:
        # adapter does not LoRA the projector (modules_to_save case) — verified separately
        W, b = base_W.astype("float32"), base_b.astype("float32")
        merged = False

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, W=W.astype("float32"), b=b.astype("float32"))
    return {"merged_lora": merged, "alpha": float(alpha), "r": int(r),
            "W_shape": list(W.shape), "b_shape": list(b.shape)}
