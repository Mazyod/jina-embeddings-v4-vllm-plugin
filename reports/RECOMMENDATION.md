# Jina v4 multi-vector multimodal embeddings on vLLM — Feasibility Verdict & Production Recommendation

**Date:** 2026-06-04 · **Status:** Feasible. Recommend shipping **Variant B**. · GPU: Modal A10G · vLLM **0.22.0**

---

## 1. Verdict

**It works.** vLLM can serve Jina Embeddings v4 **multi-vector** (128-dim/token, ColBERT-style),
**multimodal** (text + image) embeddings over HTTP, and the output matches the canonical Jina
reference to **bf16 precision** (per-token cosine ≈ 0.999 for text; ≈ 0.992–0.997 for images).
The recommended production architecture is **Variant B** — a thin FastAPI service wrapping an
in-process vLLM pooling engine, returning final 128-dim multivectors.

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
| **variant_b** (FastAPI + in-process engine) | ✅ cos_mean 0.999, aligned | ✅ cos_mean 0.992–0.997, aligned | **faithful for both modalities** |
| variant_a (stock `vllm serve` /pooling + client projection) | ✅ cos_mean 0.999, aligned | ⚠️ MISALIGNED | text faithful; images get +11 chat-template tokens |
| variant_c (in-vLLM plugin) | not built (R5 time-box) | — | see `reports/variant_c.md`; B makes it unnecessary |

**The "loss of information" question is answered:** the projection *site* does not change parity.
Variant A serializes the raw 2048-dim hidden states over JSON and still hits cos_mean 0.999 — so
neither client-side nor server-side projection loses meaningful precision. The dominant (and only
material) gap is the **bf16 floor**: the canonical Jina `encode_*` output is itself bf16, and vLLM's
backbone is bf16 with different kernels, giving ~1–2% per-element wobble while direction stays ≈0.999.

## 4. Variant comparison

| | A — stock serve + client projection | **B — FastAPI + in-process engine** | C — in-vLLM plugin |
|---|---|---|---|
| Parity (text) | cos_mean 0.999 | **cos_mean 0.999** | ≈0.999 (expected) |
| Multimodal over HTTP | needs custom chat template (token mismatch) | **faithful out of the box** | faithful (if built) |
| Wire payload | raw 2048/token (large) | **final 128/token (16× smaller)** | 128/token |
| Client complexity | high (carries projector + post-proc) | **none (clean JSON contract)** | none |
| Server complexity | none (stock vLLM) | **moderate (≈80 LOC)** | high (plugin, v1 internals) |
| Version-fragility | low | **low** | high (vLLM v1 pooler API) |

## 5. Recommendation

**Ship Variant B.** It is faithful for text *and* images, exposes a clean `{multivectors, token_ids}`
JSON contract (clients stay dumb), sends 16× less data than A, and carries low version risk. It is
~80 lines on top of `vllm serve` deps (`src/jinav4_vllm/modal_app/serve_b.py`).

- **Variant A** is a fine fallback only if a fully-stock vLLM image is a hard requirement *and* the
  workload is text-only (its multimodal HTTP path needs a custom chat template to match Jina's image
  prompt — otherwise image token sequences differ by the chat-template wrapper).
- **Variant C** offers no measurable parity benefit over B (serialization is already shown harmless)
  at the highest complexity; pursue only if native in-engine 128-dim output is independently required
  (recipe in `reports/variant_c.md`).

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

## 7. Suggested follow-ups (not blocking)

1. Throughput/latency benchmark of B at production batch sizes (not measured here).
2. Add `text-matching` / `code` adapters if needed (same recipe, different merged projector).
3. Persisted vLLM torch.compile cache + GPU memory snapshot to cut cold start (~40s today).
4. Service hardening: auth, request size limits, batching endpoint, autoscaling tuning.
