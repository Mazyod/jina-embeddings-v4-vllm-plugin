"""HF transformers ground truth. Writes /artifacts/reference/<id>.npz (mv [n,128], token_ids)."""
from __future__ import annotations
import modal
from jinav4_vllm.modal_app.app import app, ref_image, GPU, COMMON, ART, artifacts


def _mv_to_np(mv):
    """Convert a per-token multivector (torch tensor, possibly bf16) to float32 numpy."""
    import numpy as np
    if hasattr(mv, "detach"):
        return mv.detach().float().cpu().numpy().astype(np.float32)
    return np.asarray(mv, dtype=np.float32)


def _activate_retrieval(model):
    for setter in ("set_adapter", "set_task"):
        if hasattr(model, setter):
            try:
                getattr(model, setter)("retrieval")
            except Exception:
                pass


@app.function(image=ref_image, gpu=GPU, timeout=2400, **COMMON)
def reference_text():
    import sys; sys.path.insert(0, "/root")
    import os, numpy as np, torch
    from transformers import AutoModel, AutoProcessor
    from jinav4_vllm.common.probes import TEXT_PROBES, build_text_prompt
    from jinav4_vllm.common.artifacts import save_artifact

    model = AutoModel.from_pretrained("jinaai/jina-embeddings-v4", trust_remote_code=True,
                                      torch_dtype=torch.float32).eval()
    processor = AutoProcessor.from_pretrained("jinaai/jina-embeddings-v4", trust_remote_code=True)
    _activate_retrieval(model)

    os.makedirs(f"{ART}/reference", exist_ok=True)
    results = {}
    for p in TEXT_PROBES:
        # Ground-truth values from the official high-level API (authoritative).
        mv = model.encode_text(texts=[p.text], task="retrieval",
                               prompt_name=p.kind, return_multivector=True)[0]
        mv = _mv_to_np(mv)
        # Token ids from the identical prompt string (for alignment vs vLLM).
        prompt = build_text_prompt(p.text, p.kind)
        ids = np.asarray(processor.tokenizer(prompt, add_special_tokens=True)["input_ids"], dtype=np.int64)
        # Sanity: high-level multivector row count must equal our tokenization length.
        assert mv.shape[0] == ids.shape[0], (
            f"{p.id}: mv rows {mv.shape[0]} != token len {ids.shape[0]} "
            "(prompt/tokenization mismatch — fix prompt_name/template before trusting parity)")
        save_artifact(f"{ART}/reference/{p.id}.npz", mv, ids)
        results[p.id] = list(mv.shape)
    artifacts.commit()
    print(results)
    return results


@app.function(image=ref_image, gpu=GPU, timeout=2400, **COMMON)
def reference_image():
    import sys; sys.path.insert(0, "/root")
    import os, numpy as np, torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    from jinav4_vllm.common.probes import IMAGE_PROBES, build_image_prompt
    from jinav4_vllm.common.artifacts import save_artifact
    from jinav4_vllm.common.imaging import mm_processor_kwargs

    # Image fidelity (min/max pixels) from env; must match the served/offline side for parity.
    mm_kw = mm_processor_kwargs()
    model = AutoModel.from_pretrained("jinaai/jina-embeddings-v4", trust_remote_code=True,
                                      torch_dtype=torch.float32).eval()
    processor = AutoProcessor.from_pretrained("jinaai/jina-embeddings-v4", trust_remote_code=True,
                                              **mm_kw)
    _activate_retrieval(model)

    os.makedirs(f"{ART}/reference", exist_ok=True)
    results = {}
    for p in IMAGE_PROBES:
        img = Image.open(f"/root/data/probes/{os.path.basename(p.path)}").convert("RGB")
        try:
            mv = model.encode_image(images=[img], task="retrieval", return_multivector=True, **mm_kw)[0]
        except TypeError:
            mv = model.encode_image(images=[img], task="retrieval", return_multivector=True)[0]
        mv = _mv_to_np(mv)
        # Token ids via the identical prompt + processor (vision tokens expand here).
        proc = processor(text=[build_image_prompt()], images=[img], return_tensors="pt", **mm_kw)
        ids = np.asarray(proc["input_ids"][0].cpu().numpy(), dtype=np.int64)
        assert mv.shape[0] == ids.shape[0], (
            f"{p.id}: image mv rows {mv.shape[0]} != token len {ids.shape[0]}")
        save_artifact(f"{ART}/reference/{p.id}.npz", mv, ids)
        results[p.id] = list(mv.shape)
    artifacts.commit()
    print(results)
    return results
