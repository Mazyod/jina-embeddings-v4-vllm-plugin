"""Variant A: stock `vllm serve --runner pooling` exposed via Modal web_server.

The server returns raw [n,2048] per-token hidden states from /pooling; the client applies the
retrieval projector locally. Also exercises the native multimodal /pooling HTTP path (R1).

Deploy:  uv run modal deploy src/jinav4_vllm/modal_app/serve_a.py
"""
from __future__ import annotations
import modal
from jinav4_vllm.modal_app.app import app, vllm_image, GPU, COMMON

VLLM_PORT = 8000
VLLM_MODEL = "jinaai/jina-embeddings-v4-vllm-retrieval"


@app.function(image=vllm_image, gpu=GPU, timeout=3600, scaledown_window=600, **COMMON)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=VLLM_PORT, startup_timeout=600)
def serve_a():
    import subprocess, json, os
    mm = {}
    if os.environ.get("JINA_IMAGE_MIN_PIXELS"):
        mm["min_pixels"] = int(os.environ["JINA_IMAGE_MIN_PIXELS"])
    if os.environ.get("JINA_IMAGE_MAX_PIXELS"):
        mm["max_pixels"] = int(os.environ["JINA_IMAGE_MAX_PIXELS"])
    cmd = [
        "vllm", "serve", VLLM_MODEL,
        "--runner", "pooling",
        "--pooler-config.task", "token_embed",
        "--served-model-name", "jina-v4",
        "--host", "0.0.0.0", "--port", str(VLLM_PORT),
        "--max-model-len", "4096",
    ] + (["--mm-processor-kwargs", json.dumps(mm)] if mm else [])
    # Pass argv as a list (no shell) so flag values keep their exact form.
    subprocess.Popen(cmd)
