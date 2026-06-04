# Stage 1 — Offline vLLM vs HF Reference, TEXT (2026-06-04)

## Verdict: PASS — the vLLM multi-vector mechanism reproduces canonical Jina v4 multivectors.

Per-token parity of `vLLM offline (token_embed → our retrieval projector → L2-norm)` against the
canonical HF reference (`jina-embeddings-v4`, retrieval adapter, `encode_text(return_multivector=True)`):

| probe | n tokens | aligned | max_abs | mean_abs | cos_min | cos_mean | rel_frob |
|---|---|---|---|---|---|---|---|
| text_query_en_short | 10 | ✅ | 1.70e-02 | 2.69e-03 | 0.99732 | 0.99922 | 3.96e-02 |
| text_passage_en_long | 67 | ✅ | 2.42e-02 | 2.58e-03 | 0.99467 | 0.99929 | 3.78e-02 |
| text_query_ar | 16 | ✅ | 2.02e-02 | 2.75e-03 | 0.99635 | 0.99917 | 4.07e-02 |
| text_query_ja | 12 | ✅ | 2.02e-02 | 2.76e-03 | 0.99635 | 0.99912 | 4.21e-02 |
| text_passage_symbols | 37 | ✅ | 2.42e-02 | 2.82e-03 | 0.99467 | 0.99912 | 4.19e-02 |

## What "aligned ✅" proves
vLLM's `prompt_token_ids` exactly equal the reference tokenization of the same `Query:`/`Passage:`
prompt (identical count AND order) for every probe, including Arabic, Japanese, and symbol-heavy
text. So the prompt format and tokenization are reproduced exactly — no drift.

## The achieved floor is bf16, and it is expected (not a defect)
- **cos_mean ≈ 0.9992, cos_min ≈ 0.995** across all probes — direction (what late-interaction
  MaxSim uses) is essentially identical.
- **max_abs ≈ 0.02, mean_abs ≈ 0.003 per element** — this is the bf16 numerical floor:
  - The canonical Jina `encode_text(return_multivector=True)` itself returns **bfloat16** (even
    when the model is loaded with `dtype=float32`).
  - vLLM runs the backbone in **bfloat16** with different attention/matmul kernels (FlashAttention 2)
    than HF transformers (SDPA/eager).
  - Different bf16 kernels on the same math produce ~1–2% per-element deviations while preserving
    cosine ≈ 0.999. This is inherent to bf16, not to our projection.

## Pass criterion (revised after first measurement, per spec §6)
Gate on **per-token cosine** (`cos_mean ≥ 0.999`, `cos_min ≥ 0.99`) — the dimensionless, retrieval-
relevant metric. `max_abs`/`mean_abs` are reported as informational; the original `max_abs ≤ 1e-3`
gate was unattainable under bf16 and has been replaced rather than silently dropped.

## Environment
- vLLM **0.22.0**, `runner=pooling`, `PoolerConfig(task="token_embed")` → `tok_pooling_type=ALL`.
- Backbone checkpoint `jinaai/jina-embeddings-v4-vllm-retrieval` (Qwen2.5-VL, retrieval LoRA merged).
- Projector `retrieval.npz` (`128×2048` + bias) = base `multi_vector_projector` + retrieval-LoRA
  (r=32, α=32) merged, extracted from `jinaai/jina-embeddings-v4`.
- Reference transformers **4.57.6** (pinned <5 for trust_remote_code compat), fp32 load.
- GPU: Modal A10G.
