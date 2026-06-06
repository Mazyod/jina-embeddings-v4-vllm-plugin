# jina-v4-vllm-plugin

vLLM out-of-tree model plugin that makes a **stock vLLM OpenAI server** serve
**Jina Embeddings v4 multi-vector** (128-dim/token, ColBERT-style late interaction) **multimodal**
(text + image) embeddings. With the plugin installed, the server's `/pooling` endpoint returns final
L2-normalized `[n,128]` per-token multivectors directly — no proxy, no client-side projection.

It registers a `JinaV4MultiVector` architecture (Qwen2.5-VL backbone + Jina's `multi_vector_projector`
applied in-engine, mirroring vLLM's in-tree ColQwen3/ColPali pattern) via a `vllm.general_plugins`
entry point, so it loads in every vLLM process including the v1 EngineCore worker.

## Install

```bash
pip install --no-deps .          # into an image that already provides vLLM (e.g. vllm/vllm-openai)
# or build a wheel:  python -m build   /   uv build
```

`--no-deps` keeps pip from re-resolving vLLM/torch inside the official image. Pin the host vLLM
version the plugin was validated against — see `docs/COMPAT.md` in the repo.

## Use

```bash
vllm serve <jina-v4-checkpoint> \
  --runner pooling --pooler-config.task token_embed \
  --hf-overrides '{"architectures":["JinaV4MultiVector"]}' \
  --chat-template "$(python -c 'import jina_v4_vllm_plugin as p; print(p.chat_template_path())')"
```

The projector weights (`128×2048` + bias) are **not** in the vLLM checkpoint; the plugin loads them
at startup from `JINA_MV_PROJECTOR` (default `/artifacts/projector/retrieval.npz`), or from the
checkpoint itself if baked in. Full runbook: `deploy/DEPLOY.md`.
