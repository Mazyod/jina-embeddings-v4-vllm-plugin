"""Variant B: FastAPI + in-process vLLM engine; returns final [n,128] multivectors.

Production-shaped self-contained service. Runs the vLLM pooling engine in-process and applies
the retrieval projector + L2-norm server-side, so clients receive final multivectors. This also
sidesteps R1 (undocumented multimodal /pooling over HTTP) by controlling prompt construction.

Deploy:  uv run modal deploy src/jinav4_vllm/modal_app/serve_b.py

NOTE: no `from __future__ import annotations` here — FastAPI must see the real Pydantic model
class as the body-param annotation (stringized annotations + a locally-defined model make FastAPI
mis-route the body as a query param -> 422).
"""
import modal
from jinav4_vllm.modal_app.app import app, vllm_image, GPU, COMMON, ART

# vLLM already bundles fastapi/uvicorn/pydantic (its own OpenAI server deps), and we cannot add
# a build step after add_local_dir, so reuse the vLLM image directly.
web_image = vllm_image

VLLM_MODEL = "jinaai/jina-embeddings-v4-vllm-retrieval"


@app.cls(image=web_image, gpu=GPU, timeout=3600, scaledown_window=600,
         min_containers=0, **COMMON)
@modal.concurrent(max_inputs=8)
class VariantB:
    @modal.enter()
    def load(self):
        import sys; sys.path.insert(0, "/root")
        import numpy as np
        from vllm import LLM
        from vllm.config import PoolerConfig
        from jinav4_vllm.multivector.core import to_multivector
        self.np = np
        self.to_multivector = to_multivector
        self.llm = LLM(model=VLLM_MODEL, runner="pooling",
                       pooler_config=PoolerConfig(task="token_embed"),
                       max_model_len=4096, gpu_memory_utilization=0.85)
        proj = np.load(f"{ART}/projector/retrieval.npz")
        self.W, self.b = proj["W"], proj["b"]

    def _encode(self, prompt_obj):
        out = self.llm.encode([prompt_obj], pooling_task="token_embed")[0]
        hidden = out.outputs.data
        hidden = self.np.asarray(
            hidden.detach().cpu().numpy() if hasattr(hidden, "detach") else hidden,
            dtype=self.np.float32)
        mv = self.to_multivector(hidden, self.W, self.b)          # [n,128] server-side
        return mv, [int(t) for t in out.prompt_token_ids]

    @modal.asgi_app()
    def web(self):
        import base64, io
        from fastapi import FastAPI, Request
        from PIL import Image
        from jinav4_vllm.common.probes import build_text_prompt, build_image_prompt

        api = FastAPI()

        @api.post("/embed/text")
        async def embed_text(request: Request):
            body = await request.json()
            mv, ids = self._encode(build_text_prompt(body["text"], body.get("kind", "query")))
            return {"dim": 128, "tokens": mv.shape[0], "multivectors": mv.tolist(), "token_ids": ids}

        @api.post("/embed/image")
        async def embed_image(request: Request):
            body = await request.json()
            img = Image.open(io.BytesIO(base64.b64decode(body["image_b64"]))).convert("RGB")
            mv, ids = self._encode({"prompt": build_image_prompt(), "multi_modal_data": {"image": img}})
            return {"dim": 128, "tokens": mv.shape[0], "multivectors": mv.tolist(), "token_ids": ids}

        return api
