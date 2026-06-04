"""vLLM offline engine. Writes /artifacts/offline/<id>.npz (raw mv [n,2048], token_ids)."""
from __future__ import annotations
import modal
from jinav4_vllm.modal_app.app import app, vllm_image, GPU, COMMON, ART, artifacts

VLLM_MODEL = "jinaai/jina-embeddings-v4-vllm-retrieval"


def _build_engine(max_model_len):
    from vllm import LLM
    from vllm.config import PoolerConfig
    return LLM(model=VLLM_MODEL, runner="pooling",
               pooler_config=PoolerConfig(task="token_embed"),
               max_model_len=max_model_len, gpu_memory_utilization=0.85)


def _to_np(data):
    import numpy as np
    return np.asarray(data.detach().cpu().numpy() if hasattr(data, "detach") else data,
                      dtype=np.float32)


@app.function(image=vllm_image, gpu=GPU, timeout=2400, **COMMON)
def offline_text():
    import sys; sys.path.insert(0, "/root")
    import os, numpy as np
    from vllm.inputs.data import TextPrompt
    from jinav4_vllm.common.probes import TEXT_PROBES, build_text_prompt
    from jinav4_vllm.common.artifacts import save_artifact

    llm = _build_engine(2048)
    os.makedirs(f"{ART}/offline", exist_ok=True)
    prompts = [TextPrompt(prompt=build_text_prompt(p.text, p.kind)) for p in TEXT_PROBES]
    outputs = llm.encode(prompts)
    results = {}
    for p, out in zip(TEXT_PROBES, outputs):
        hidden = _to_np(out.outputs.data)            # [n, 2048]
        ids = np.asarray(out.prompt_token_ids, dtype=np.int64)
        save_artifact(f"{ART}/offline/{p.id}.npz", hidden, ids)
        results[p.id] = list(hidden.shape)
    artifacts.commit()
    print(results)
    return results


@app.function(image=vllm_image, gpu=GPU, timeout=2400, **COMMON)
def offline_image():
    import sys; sys.path.insert(0, "/root")
    import os, numpy as np
    from PIL import Image
    from vllm.inputs.data import TextPrompt
    from jinav4_vllm.common.probes import IMAGE_PROBES, build_image_prompt
    from jinav4_vllm.common.artifacts import save_artifact

    llm = _build_engine(4096)
    os.makedirs(f"{ART}/offline", exist_ok=True)
    results = {}
    for p in IMAGE_PROBES:
        img = Image.open(f"/root/data/probes/{os.path.basename(p.path)}").convert("RGB")
        out = llm.encode([TextPrompt(prompt=build_image_prompt(), multi_modal_data={"image": img})])[0]
        hidden = _to_np(out.outputs.data)
        ids = np.asarray(out.prompt_token_ids, dtype=np.int64)
        save_artifact(f"{ART}/offline/{p.id}.npz", hidden, ids)
        results[p.id] = list(hidden.shape)
    artifacts.commit()
    print(results)
    return results
