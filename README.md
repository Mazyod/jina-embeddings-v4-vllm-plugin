# Jina v4 multi-vector multimodal embeddings on vLLM

Serve **Jina Embeddings v4** **multi-vector** (128-dim/token, ColBERT-style late interaction)
**multimodal** (text + image) embeddings from a **stock vLLM OpenAI server**. A small out-of-tree
vLLM model plugin (`JinaV4MultiVector`) applies Jina's multi-vector projection **in-engine**, so the
standard server's `/pooling` endpoint returns final L2-normalized `[n,128]` per-token vectors —
matching canonical Jina output to bf16 precision (per-token cosine ≈ 0.999 text, ≈ 0.992–0.997 image).

This repo is dedicated to **maintaining the plugin** and **demonstrating/validating** how it works.

## Why a plugin?
vLLM's Jina-v4 checkpoints omit the 128-dim `multi_vector_projector`; stock pooling only yields the
raw 2048-dim hidden states. The plugin mirrors vLLM's in-tree ColQwen3/ColPali pattern (Qwen2.5-VL
backbone + a ColBERT projection + `pooler_for_token_embed`) to produce true multi-vectors natively.

## Layout

| Path | What |
|---|---|
| **`src/jinav4_vllm/vllm_plugin/`** | **The plugin** — `JinaV4MultiVector` model + entry-point registration + image chat template. The maintained core. |
| **`deploy/`** | Production hand-off: `DEPLOY.md` runbook, `Dockerfile` (extends official `vllm/vllm-openai`), `bake_checkpoint.py` (drop-in checkpoint builder). |
| `src/jinav4_vllm/projector/` | Extract the retrieval-effective projector (base + retrieval-LoRA merged) → `retrieval.npz`. |
| `src/jinav4_vllm/multivector/`, `eval/`, `common/` | Pure-NumPy projection math, parity metrics, probes, image-fidelity config (unit-tested). |
| `src/jinav4_vllm/modal_app/` | Modal harnesses used to validate parity on GPU (reference vs offline vs the 3 serving variants) + the `bake_checkpoint` job + recon diagnostics. |
| `reports/` | Evidence: `RECOMMENDATION.md`, `stage1_text.md`, `stage1_image.md`, `variant_c.md`, parity tables. |
| `docs/superpowers/` | Original design spec + implementation plan. |

## Quickstart (production — Variant C)

```bash
# 1) build the serving image (official vLLM image + plugin)
docker build -f deploy/Dockerfile --build-arg VLLM_TAG=v0.22.0 -t jina-v4-mv-vllm:0.22.0 .

# 2) bake a drop-in checkpoint (projector + architecture + chat template; raise --max-pixels for fidelity)
python deploy/bake_checkpoint.py --out ./jina-v4-mv-baked --max-pixels 3211264

# 3) serve — nothing but the pooling flags
docker run --gpus all -p 8000:8000 -v $PWD/jina-v4-mv-baked:/m jina-v4-mv-vllm:0.22.0 \
  /m --runner pooling --pooler-config.task token_embed --served-model-name jina-v4

# 4) query  (per-token multivectors on /pooling)
curl -s localhost:8000/pooling -H 'Content-Type: application/json' \
  -d '{"model":"jina-v4","input":["Query: hello world"]}'
```

Full runbook (both deployment modes, image requests, MaxSim scoring, image fidelity, ops notes):
**`deploy/DEPLOY.md`**. Verdict & variant comparison: **`reports/RECOMMENDATION.md`**.

## Validate / demo (Modal GPU)

Pure-logic tests run locally (`uv run pytest`). GPU parity runs on Modal:

```bash
uv run modal run src/jinav4_vllm/modal_app/app.py::extract_projector   # build retrieval.npz
uv run modal run src/jinav4_vllm/modal_app/reference.py::reference_text # + reference_image
uv run modal run src/jinav4_vllm/modal_app/offline.py::offline_text     # + offline_image
uv run modal deploy src/jinav4_vllm/modal_app/serve_c.py                # native server (serve_c / serve_baked)
uv run python -m jinav4_vllm.eval.report                                # element-wise parity table
```

## Image fidelity
Qwen2.5-VL uses dynamic resolution; the checkpoint default caps it low. Control via min/max pixels —
bake into the checkpoint (`--max-pixels`) or set `JINA_IMAGE_MIN_PIXELS` / `JINA_IMAGE_MAX_PIXELS`.
See `deploy/DEPLOY.md` § Image fidelity. Keep reference and served sides equal or per-token parity
breaks (token counts change with resolution).

## Versioning
The plugin touches vLLM internals — **pin the vLLM version** and re-validate on upgrades (the
`recon_*` functions in `src/jinav4_vllm/modal_app/app.py` regenerate the needed API facts cheaply).
