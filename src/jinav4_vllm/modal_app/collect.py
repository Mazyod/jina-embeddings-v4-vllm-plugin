"""Pull embeddings from a deployed vLLM server and save artifacts/served/<id>.npz for parity.

This is the client-side validation helper (no Modal/GPU needed): it hits the deployed `/pooling`
endpoint via the JinaV4Client SDK, then `jinav4_vllm.eval.report` compares artifacts/served against
artifacts/reference (HF ground truth) and artifacts/offline.

    python -m jinav4_vllm.modal_app.collect <BASE_URL> [model]          # capture for parity
    python -m jinav4_vllm.modal_app.collect <BASE_URL> [model] --smoke  # /pooling contract check
"""
from __future__ import annotations

import os
import sys

import numpy as np

from jinav4_vllm.client import JinaV4Client
from jinav4_vllm.common.artifacts import save_artifact
from jinav4_vllm.common.probes import IMAGE_PROBES, TEXT_PROBES

OUT_DIR = "artifacts/served"


def collect_served(base_url: str, model: str = "jina-v4") -> None:
    """Hit the deployed /pooling endpoint for every probe; save final [n,128] multivectors."""
    client = JinaV4Client(base_url, model=model)
    os.makedirs(OUT_DIR, exist_ok=True)
    for p in TEXT_PROBES:
        mv = client.embed_text(p.text, p.kind)
        save_artifact(f"{OUT_DIR}/{p.id}.npz", mv, np.arange(mv.shape[0], dtype=np.int64))
    for p in IMAGE_PROBES:
        try:
            mv = client.embed_image(p.path)
            save_artifact(f"{OUT_DIR}/{p.id}.npz", mv, np.arange(mv.shape[0], dtype=np.int64))
        except Exception as e:  # noqa: BLE001 — report and continue to the next probe
            print(f"[served] image {p.id} failed: {e}")
    print(f"wrote {OUT_DIR}/ from {base_url}")


def smoke(base_url: str, model: str = "jina-v4") -> dict:
    """Liveness + contract check: /pooling returns L2-normalized 128-dim vectors for text and image."""
    client = JinaV4Client(base_url, model=model)
    t = client.embed_text("hello world", "query")
    assert t.ndim == 2 and t.shape[1] == 128, f"text dim {t.shape} != [*,128]"
    assert np.allclose(np.linalg.norm(t, axis=1), 1.0, atol=1e-3), "text vectors not L2-normalized"
    img = client.embed_image(IMAGE_PROBES[0].path)
    assert img.ndim == 2 and img.shape[1] == 128, f"image dim {img.shape} != [*,128]"
    result = {"text_shape": list(t.shape), "image_shape": list(img.shape)}
    print("SMOKE PASS", result)
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m jinav4_vllm.modal_app.collect <BASE_URL> [model] [--smoke]")
    base = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "jina-v4"
    (smoke if "--smoke" in sys.argv else collect_served)(base, model)
