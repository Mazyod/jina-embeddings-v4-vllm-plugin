"""Hit a deployed variant endpoint and write artifacts/<variant>/<id>.npz locally.

Usage:
  # Variant B (server returns final [n,128]):
  python -m jinav4_vllm.modal_app.client variant_b <BASE_URL> /embed/text /embed/image
  # Variant A (stock vllm serve /pooling returns [n,2048]; client projects):
  python -c "from jinav4_vllm.modal_app.client import collect_variant_a; collect_variant_a('<URL>')"
  # Variant C (custom pooler /pooling returns [n,128]):
  python -c "from jinav4_vllm.modal_app.client import collect_variant_c; collect_variant_c('<URL>')"
"""
from __future__ import annotations
import base64, os, sys
import numpy as np
import requests
from jinav4_vllm.common.probes import TEXT_PROBES, IMAGE_PROBES, build_text_prompt
from jinav4_vllm.common.artifacts import save_artifact


def collect(base_url: str, variant: str, text_path: str, image_path: str):
    """Variant B style: server returns final multivectors + token_ids via a clean JSON contract."""
    os.makedirs(f"artifacts/{variant}", exist_ok=True)
    for p in TEXT_PROBES:
        r = requests.post(f"{base_url}{text_path}", json={"text": p.text, "kind": p.kind}, timeout=180)
        r.raise_for_status(); d = r.json()
        save_artifact(f"artifacts/{variant}/{p.id}.npz",
                      np.asarray(d["multivectors"], np.float32), np.asarray(d["token_ids"], np.int64))
    for p in IMAGE_PROBES:
        b64 = base64.b64encode(open(p.path, "rb").read()).decode()
        r = requests.post(f"{base_url}{image_path}", json={"image_b64": b64}, timeout=180)
        r.raise_for_status(); d = r.json()
        save_artifact(f"artifacts/{variant}/{p.id}.npz",
                      np.asarray(d["multivectors"], np.float32), np.asarray(d["token_ids"], np.int64))
    print(f"wrote artifacts/{variant}/ from {base_url}")


def _pooling_text(base_url: str, model: str, prompt: str) -> np.ndarray:
    r = requests.post(f"{base_url}/pooling", json={"model": model, "input": [prompt]}, timeout=180)
    r.raise_for_status()
    return np.asarray(r.json()["data"][0]["data"], np.float32)


def _pooling_image(base_url: str, model: str, b64: str) -> np.ndarray:
    msg = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": "Describe the image."}]}]
    r = requests.post(f"{base_url}/pooling", json={"model": model, "messages": msg}, timeout=180)
    r.raise_for_status()
    return np.asarray(r.json()["data"][0]["data"], np.float32)


def collect_variant_a(base_url: str, model: str = "jina-v4"):
    """Variant A: stock /pooling returns raw [n,2048]; client applies the projector locally."""
    from jinav4_vllm.multivector.core import to_multivector
    proj = np.load("artifacts/projector/retrieval.npz"); W, b = proj["W"], proj["b"]
    os.makedirs("artifacts/variant_a", exist_ok=True)
    for p in TEXT_PROBES:
        hidden = _pooling_text(base_url, model, build_text_prompt(p.text, p.kind))
        ids = np.arange(hidden.shape[0], dtype=np.int64)   # /pooling drops token_ids -> count-only align
        save_artifact(f"artifacts/variant_a/{p.id}.npz", to_multivector(hidden, W, b), ids)
    for p in IMAGE_PROBES:
        b64 = base64.b64encode(open(p.path, "rb").read()).decode()
        try:
            hidden = _pooling_image(base_url, model, b64)
            ids = np.arange(hidden.shape[0], dtype=np.int64)
            save_artifact(f"artifacts/variant_a/{p.id}.npz", to_multivector(hidden, W, b), ids)
        except requests.HTTPError as e:
            print(f"[R1] native multimodal /pooling failed for {p.id}: {e} "
                  "-> variant A image-over-HTTP unsupported without a custom chat template (documented).")
    print("wrote artifacts/variant_a/")


def collect_variant_c(base_url: str, model: str = "jina-v4"):
    """Variant C: custom in-vLLM pooler -> /pooling returns final [n,128] (no client projection)."""
    os.makedirs("artifacts/variant_c", exist_ok=True)
    for p in TEXT_PROBES:
        mv = _pooling_text(base_url, model, build_text_prompt(p.text, p.kind))
        save_artifact(f"artifacts/variant_c/{p.id}.npz", mv, np.arange(mv.shape[0], dtype=np.int64))
    for p in IMAGE_PROBES:
        b64 = base64.b64encode(open(p.path, "rb").read()).decode()
        try:
            mv = _pooling_image(base_url, model, b64)
            save_artifact(f"artifacts/variant_c/{p.id}.npz", mv, np.arange(mv.shape[0], dtype=np.int64))
        except requests.HTTPError as e:
            print(f"[variant_c] image over /pooling failed for {p.id}: {e}")
    print("wrote artifacts/variant_c/")


if __name__ == "__main__":
    collect(base_url=sys.argv[2], variant=sys.argv[1], text_path=sys.argv[3], image_path=sys.argv[4])
