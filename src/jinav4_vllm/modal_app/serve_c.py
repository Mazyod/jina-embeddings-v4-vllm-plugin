"""Variant C: STOCK `vllm serve` OpenAI server that natively returns 128-dim multivectors.

The JinaV4MultiVector plugin (installed in vllm_plugin_image) registers an out-of-tree model that
applies Jina's multi_vector_projector + L2-norm inside the engine. So the standard vLLM OpenAI
server's `/pooling` endpoint returns final `[n,128]` per-token multivectors directly — no proxy,
no client-side math. This is the "unify on the vLLM OpenAI image" route.

Deploy:  uv run modal deploy src/jinav4_vllm/modal_app/serve_c.py
"""
from __future__ import annotations
import modal
from jinav4_vllm.modal_app.app import app, vllm_plugin_image, GPU, COMMON

VLLM_PORT = 8000
VLLM_MODEL = "jinaai/jina-embeddings-v4-vllm-retrieval"


BAKED_CKPT = "/artifacts/jina-v4-mv-baked"  # produced by app.py::bake_checkpoint


@app.function(image=vllm_plugin_image, gpu=GPU, timeout=3600, scaledown_window=600, **COMMON)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=VLLM_PORT, startup_timeout=900)
def serve_c():
    import subprocess
    # The plugin reads the projector from /artifacts/projector/retrieval.npz (mounted via COMMON).
    cmd = [
        "vllm", "serve", VLLM_MODEL,
        "--runner", "pooling",
        "--pooler-config.task", "token_embed",
        "--hf-overrides", '{"architectures": ["JinaV4MultiVector"]}',
        "--served-model-name", "jina-v4",
        "--host", "0.0.0.0", "--port", str(VLLM_PORT),
        "--max-model-len", "4096",
        # Custom chat template emits Jina's exact image prompt so multimodal /pooling token
        # sequences match the canonical reference (avoids the default-template wrapper tokens).
        "--chat-template", "/opt/jina_plugin/jina_image_chat_template.jinja",
    ]
    subprocess.Popen(cmd)


@app.function(image=vllm_plugin_image, gpu=GPU, timeout=3600, scaledown_window=600, **COMMON)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=VLLM_PORT, startup_timeout=900)
def serve_baked():
    """Mode B drop-in: serve the baked checkpoint with NO --hf-overrides/--chat-template/env var.

    architectures, the projector tensors, and the chat template all live inside the checkpoint.
    """
    import subprocess
    cmd = [
        "vllm", "serve", BAKED_CKPT,
        "--runner", "pooling",
        "--pooler-config.task", "token_embed",
        "--served-model-name", "jina-v4",
        "--host", "0.0.0.0", "--port", str(VLLM_PORT),
        "--max-model-len", "4096",
    ]
    subprocess.Popen(cmd)
