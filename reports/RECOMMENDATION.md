# Jina v4 multi-vector multimodal embeddings on vLLM — Feasibility Verdict & Production Recommendation

**Date:** 2026-06-05 · **Status:** Feasible. Recommend **Variant C** (native vLLM OpenAI server) to unify infra; Variant B as fallback. · GPU: Modal A10G · vLLM **0.22.0**

---

## 1. Verdict

**It works — including the native route.** vLLM can serve Jina Embeddings v4 **multi-vector**
(128-dim/token, ColBERT-style), **multimodal** (text + image) embeddings over HTTP, matching the
canonical Jina reference to **bf16 precision** (per-token cosine ≈ 0.999 for text; ≈ 0.992–0.997 for
images). **All three projection sites were built and proven**, and crucially the **stock `vllm serve`
OpenAI server (Variant C)** now returns final 128-dim multivectors directly from `/pooling` — via a
small out-of-tree model plugin + a custom chat template. Recommended choice: **Variant C** for teams
already serving LLMs on the vLLM OpenAI image (unifies infrastructure); **Variant B** (FastAPI +
in-process engine) as a no-plugin fallback.

## 2. How it works (the mechanism we proved)

vLLM has no built-in path for Jina v4 multivectors, so the pipeline is:

1. Serve the adapter-merged checkpoint `jinaai/jina-embeddings-v4-vllm-retrieval` in pooling mode:
   `runner="pooling"`, `PoolerConfig(task="token_embed")` → vLLM returns **raw `[n,2048]`** per-token
   hidden states (`tok_pooling_type=ALL`).
2. Apply the **retrieval-effective `multi_vector_projector`** (a `128×2048` linear + bias) and
   per-token L2-normalize → final `[n,128]` multivectors.
   - This projector is **absent from the vLLM checkpoints**; we extract it from the main
     `jinaai/jina-embeddings-v4` repo and merge the retrieval LoRA (r=32, α=32) into it
     (`projector/` + verified by reproducing the model's own projector).
3. For images, prompt with Jina's exact template
   `"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|>\n"`
   + the raw image; for text, `"Query: …"` / `"Passage: …"`.

## 3. Parity results (element-wise vs canonical HF reference)

Reference = `jinaai/jina-embeddings-v4`, retrieval adapter, `encode_*(return_multivector=True)`.
Gate = aggregate per-token **cos_mean ≥ 0.99** (direction is what late-interaction MaxSim uses).
Full numbers in `reports/parity.md`, `reports/stage1_text.md`, `reports/stage1_image.md`.

| path | text (5 probes) | image (2 probes) | notes |
|---|---|---|---|
| offline (in-process baseline) | ✅ cos_mean 0.999, aligned | ✅ cos_mean 0.992–0.997, aligned | token_ids match reference exactly |
| **variant_c** (stock `vllm serve` + plugin) | ✅ cos_mean 0.999, aligned | ✅ cos_mean 0.992–0.997, aligned | **native OpenAI server returns [n,128]; faithful both modalities** |
| **variant_b** (FastAPI + in-process engine) | ✅ cos_mean 0.999, aligned | ✅ cos_mean 0.992–0.997, aligned | faithful for both modalities |
| variant_a (stock `vllm serve` /pooling + client projection) | ✅ cos_mean 0.999, aligned | ⚠️ MISALIGNED | text faithful; images need the same custom chat template as C |

**The "loss of information" question is answered:** the projection *site* does not change parity.
Variant A serializes the raw 2048-dim hidden states over JSON and still hits cos_mean 0.999 — so
neither client-side nor server-side projection loses meaningful precision. The dominant (and only
material) gap is the **bf16 floor**: the canonical Jina `encode_*` output is itself bf16, and vLLM's
backbone is bf16 with different kernels, giving ~1–2% per-element wobble while direction stays ≈0.999.

## 4. Variant comparison

| | A — stock serve + client projection | B — FastAPI + in-process engine | **C — stock serve + plugin** |
|---|---|---|---|
| Parity (text) | cos_mean 0.999 | cos_mean 0.999 | **cos_mean 0.999** |
| Parity (image) | n/a (misaligned) | 0.992–0.997 | **0.992–0.997** |
| Multimodal over HTTP | needs custom chat template | faithful (prompt control) | **faithful (custom chat template)** |
| Server image | stock vLLM OpenAI | custom FastAPI | **stock vLLM OpenAI** |
| Endpoint | native `/pooling` (raw 2048) | custom `/embed/*` | **native `/pooling` (128)** |
| Wire payload | raw 2048/token (large) | final 128/token | **final 128/token** |
| Client complexity | high (carries projector) | none | **none** |
| Server complexity | none | moderate (≈80 LOC) | **small plugin (model class + template)** |
| Infra unification | yes | no (separate service) | **yes (same vLLM image as other LLMs)** |
| Version-fragility | low | low | medium (out-of-tree model vs vLLM internals) |

## 5. Recommendation

**Choose Variant C** if you serve other LLMs on the vLLM OpenAI image and want one unified
serving stack: it is a stock `vllm serve` whose `/pooling` returns final L2-normalized `[n,128]`
multivectors for text *and* images, at the same parity as B (cos_mean 0.999 / 0.992–0.997). The cost
is carrying a small out-of-tree model plugin (`src/jinav4_vllm/vllm_plugin/`) that touches vLLM
internals, so pin the vLLM version and re-validate the model class on upgrades (`reports/variant_c.md`).
For a fully drop-in artifact, bake the projector into the checkpoint so no env var / `--hf-overrides`
is needed.

- **Variant B** is the fallback if you prefer not to maintain an out-of-tree vLLM model: a small
  self-contained FastAPI service (`serve_b.py`) with a clean `{multivectors, token_ids}` contract and
  no vLLM-internals coupling — but it is a separate service from your other LLMs.
- **Variant A** (stock serve, client-side projection) is viable for text-only workloads; for images
  it needs the same custom chat template as C and pushes 16× larger payloads (raw 2048-dim) over the
  wire. Empirically that serialization still preserves parity (cos_mean 0.999), so it's a valid
  text-only option but not preferred.

## 6. Operational notes for productionizing B

- **vLLM 0.22 API:** `LLM(..., runner="pooling", pooler_config=PoolerConfig(task="token_embed"))`;
  `llm.encode(prompts, pooling_task="token_embed")` (the task arg is required). CLI equivalent for
  stock serve: `--runner pooling --pooler-config.task token_embed`.
- **transformers pin:** the HF reference needs `transformers<5` (+ `torchvision`) for Jina v4's
  `trust_remote_code`; vLLM serving is unaffected (vLLM has native Qwen2.5-VL).
- **dtype:** both reference and vLLM run bf16; expect cos ≈0.999, not bit-identical. If tighter image
  parity is ever needed, see `reports/stage1_image.md` (fp32 vision tower / pinned image processor).
- **Deploys:** `modal app stop` before redeploying a served variant, or a warm container serves stale
  code (10-min scaledown window).
- **Projector artifact:** ship `retrieval.npz` (≈1 MB) alongside the service; it is the only piece not
  inside the vLLM checkpoint.
- **Variant C (native) deployment:** stock `vllm serve <ckpt> --runner pooling --pooler-config.task
  token_embed --hf-overrides '{"architectures":["JinaV4MultiVector"]}' --chat-template
  jina_image_chat_template.jinja`, with the `jina-v4-vllm-plugin` package installed and `retrieval.npz`
  reachable (default `/artifacts/projector/retrieval.npz`, or `JINA_MV_PROJECTOR`). Per-token output is
  the `/pooling` endpoint (`/v1/embeddings` only returns one pooled vector). Full recipe + a
  bake-into-checkpoint option in `reports/variant_c.md`.

## 7. Suggested follow-ups (not blocking)

1. Throughput/latency benchmark of B at production batch sizes (not measured here).
2. Add `text-matching` / `code` adapters if needed (same recipe, different merged projector).
3. Persisted vLLM torch.compile cache + GPU memory snapshot to cut cold start (~40s today).
4. Service hardening: auth, request size limits, batching endpoint, autoscaling tuning.
