# Validation — Jina v4 multi-vector multimodal on vLLM

Evidence that the `JinaV4MultiVector` plugin reproduces canonical Jina Embeddings v4 multi-vector
embeddings when served from a stock vLLM OpenAI server. Verified on **Modal A10G**, **vLLM 0.22.0**,
reference **transformers 4.57.6** (pinned `<5` for Jina's `trust_remote_code`).

**Bottom line:** the served `/pooling` output matches the canonical HF reference to **bf16
precision** — per-token cosine ≈ **0.999** (text) and ≈ **0.992–0.997** (image) — with exact
token-count alignment for both modalities. This is a faithful drop-in for canonical Jina v4 at bf16.

## How it works (the mechanism)

1. Serve the adapter-merged checkpoint `jinaai/jina-embeddings-v4-vllm-retrieval` in pooling mode
   (`--runner pooling --pooler-config.task token_embed`) → vLLM yields raw `[n,2048]` per-token
   hidden states (`tok_pooling_type = ALL`).
2. The plugin applies the **retrieval-effective `multi_vector_projector`** (`128×2048` linear + bias,
   fp32) in-engine and per-token L2-normalizes → final `[n,128]`. That projector is **absent from the
   vLLM checkpoint**; it is extracted from `jinaai/jina-embeddings-v4` with the retrieval LoRA
   (r=32, α=32) merged in (`src/jinav4_vllm/projector/`, verified by `revalidate.py::verify_projector`
   reproducing the model's own projector to `max_abs_diff < 1e-3`).
3. Text uses `"Query: …"` / `"Passage: …"`; images use Jina's exact template
   `"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|>\n"`.

The deployed server proves the same numbers as the in-process engine: the projection *site* (in
the engine, in a wrapper, or client-side) does not change parity. The only material gap is the
**bf16 floor** (below).

## Parity results (element-wise vs canonical HF reference)

Reference = `jinaai/jina-embeddings-v4`, retrieval adapter, `encode_*(return_multivector=True)`.
Gate = aggregate per-token **cos_mean ≥ 0.99** (direction is what late-interaction MaxSim uses).
Reproduce with `make reference offline parity` (and `make collect URL=…` to add the served column).

### Text — `offline` and `served` both PASS

| probe | n tokens | aligned | max_abs | mean_abs | cos_min | cos_mean |
|---|---|---|---|---|---|---|
| text_query_en_short | 10 | ✅ | 1.70e-02 | 2.69e-03 | 0.99732 | 0.99922 |
| text_passage_en_long | 67 | ✅ | 2.42e-02 | 2.58e-03 | 0.99467 | 0.99929 |
| text_query_ar | 16 | ✅ | 2.02e-02 | 2.75e-03 | 0.99635 | 0.99917 |
| text_query_ja | 12 | ✅ | 2.02e-02 | 2.76e-03 | 0.99635 | 0.99912 |
| text_passage_symbols | 37 | ✅ | 2.42e-02 | 2.82e-03 | 0.99467 | 0.99912 |

vLLM's `prompt_token_ids` exactly equal the reference tokenization (count **and** order) for every
probe — Arabic, Japanese, and symbol-heavy text included. No prompt/tokenization drift.

### Image — `offline` and `served` both PASS

| probe | n tokens | aligned | max_abs | mean_abs | cos_min | cos_mean |
|---|---|---|---|---|---|---|
| image_cat | 75 | ✅ | 1.49e-01 | 5.98e-03 | 0.81420 | 0.99206 |
| image_chart | 110 | ✅ | 7.01e-02 | 5.03e-03 | 0.96162 | 0.99663 |

Image-token counts match the reference **exactly** (cat=75, chart=110) — same patch grid, sequences
align 1:1. Every low-cosine token is `token_id=151655` (the image-pad / image-patch token); text and
special tokens are unaffected. The dip is the **bf16 vision-encoder floor** (deep ViT, FlashAttention-2
vs HF kernels), not a pipeline error — `image_cat` has 8/75 patches < 0.99 (worst 0.814), `image_chart`
4/110 (worst 0.962), while the aggregate direction stays ≥ 0.99. MaxSim is robust to a few divergent
document patches.

## Why it is bf16, not bit-exact

- The canonical Jina `encode_*(return_multivector=True)` itself returns **bfloat16** (even when the
  model is loaded `dtype=float32`).
- vLLM runs the backbone in **bfloat16** with different attention/matmul kernels (FlashAttention-2)
  than HF transformers (SDPA/eager).
- Different bf16 kernels on the same math give ~1–2% per-element wobble while preserving cosine
  ≈ 0.999. This is inherent to bf16, not to the projection.

If bit-tighter image parity were ever required (not needed for retrieval): run the vision tower in
fp32 on both sides, and/or pin identical image-processor settings — counts already match, so that
only addresses sub-patch interpolation, second-order vs the bf16 kernel gap.

## Image fidelity note

Image-token count scales with resolution (`min_pixels`/`max_pixels`; see `deploy/DEPLOY.md` §
Image fidelity). The parity check aligns per-token, so the **reference and served/offline sides must
use the same pixel bounds** — otherwise counts diverge and alignment (R2) fails. Repo defaults leave
them unset (checkpoint default) to keep this demo green.
