# Jina v4 Multi-Vector Multimodal Embeddings on vLLM — Feasibility Study & Production Design

**Date:** 2026-06-04
**Status:** Approved design — ready for implementation planning
**Type:** Production feasibility study (ships to production if successful)

---

## 1. Problem & Goal

We use **Jina Embeddings v4** to embed text and images. We want to serve it on **vLLM** (our preferred inference engine) using its **multi-vector** (late-interaction / ColBERT-style, 128-dim-per-token) output, with **multimodal** (text + image) support, exposed over an HTTP service we can run on our infra.

vLLM has no documented native path for Jina v4 multi-vector output. This study determines **how** to make it work, **proves numerical parity** against the reference implementation, and **compares three production architectures** so we can ship the best one.

This is **not a throwaway PoC**: the winning variant is intended for production.

### Success criteria
1. At least one served variant reproduces the reference multi-vector outputs within tolerance for **both** text and images (numerical parity).
2. A defensible **recommendation** among the three projection-site variants, based on measured parity **and** operational trade-offs (latency, wire size, complexity, version-fragility).
3. Reproducible repo + deployed Modal endpoint(s) + a parity report with numbers.

### Out of scope
- Curated retrieval benchmark / relevance-labelled corpus (explicitly dropped — see §6).
- Throughput/latency benchmarking as a gate (may be reported opportunistically, not required).
- `text-matching` and `code` adapters (retrieval adapter only).
- Production hardening beyond what the feasibility decision requires (auth, autoscaling tuning, observability) — noted as follow-ups.

---

## 2. Key Research Findings (the "how")

All findings verified against primary sources (HF model cards, checkpoint `config.json` / weight indices / `modeling_jina_embeddings_v4.py`, vLLM docs + example source, Modal docs).

1. **Architecture.** Jina v4 = `Qwen/Qwen2.5-VL-3B-Instruct` backbone (3.8B + ~60M per LoRA adapter), custom class `JinaEmbeddingsV4Model`. Two output heads:
   - **single-vector:** masked **mean** pool of last hidden states → 2048-dim, L2-normalized (Matryoshka-truncatable to [128,256,512,1024,2048]). No extra weights.
   - **multi-vector:** a learned `nn.Linear(2048 → 128)` (`multi_vector_projector`) applied **per token**, then per-token L2-normalize. `multi_vector_projector_dim = 128`.
2. **The vLLM checkpoints omit the projector.** `jinaai/jina-embeddings-v4-vllm-retrieval` is a stock `Qwen2_5_VLForConditionalGeneration` with the **retrieval LoRA merged into the backbone**. Weight-index diff confirms it contains **no** `multi_vector_projector` tensors. Those weights exist **only** in the main `jinaai/jina-embeddings-v4` repo (`model-00002-of-00002.safetensors`: `multi_vector_projector.weight` [128×2048], `multi_vector_projector.bias` [128]).
3. **No published multi-vector recipe.** Jina's HF cards and vLLM's own `examples/pooling/token_embed/jina_embeddings_v4_offline.py` all stop at **single-vector dense pooling** (sum/mean → normalize on raw 2048-dim hidden states). Producing true 128-dim multi-vectors is the novel work of this study.
4. **The projector is itself LoRA-adapted per task.** The retrieval adapter (`r=32, alpha=32`) targets `q/k/v/o/gate/up/down_proj` **and** `single_vector_projector`/`multi_vector_projector` (with `exclude_modules: .*visual.*`). Therefore the projector we apply must be the **retrieval-effective** projector = base projector + retrieval-LoRA delta, merged — **not** the bare base projector.
5. **vLLM mechanism for per-token output.**
   - Offline: `LLM(model=..., runner="pooling", pooler_config=PoolerConfig(task="token_embed"))` → `output.outputs.data` is `[num_tokens, 2048]`; `output.prompt_token_ids` gives the token ids for slicing.
   - Served (HTTP): per-token output comes from the **`/pooling`** endpoint (response `data` is a nested `[num_tokens, dim]` list), **not** `/v1/embeddings` (single vector only). Start with `vllm serve <model> --runner pooling --pooler-config.task token_embed`.
   - Note: `--task embed` is deprecated → `--runner pooling`. Directly forcing `PoolerConfig(pooling_type="ALL")` for an `embed` task is a known regression on some versions (vLLM issue #25165); use `task="token_embed"`.
6. **Multimodal over HTTP.** Images are passed as OpenAI chat-style `messages` with `image_url` content (base64 data URI supported). vLLM's `/pooling` accepts `messages` for multimodal pooling models (confirmed via the ColQwen3 token_embed online example). **Jina v4 multimodal over `/pooling` is not officially documented** → must be validated early (Risk R1).
7. **Prompt format.** Text query: `"Query: {text}"`; passage: `"Passage: {text}"`. Image (offline): prompt string `"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|>\n"` + `multi_modal_data={"image": image}`. Special token ids: `vision_start=151652`, `vision_end=151653`, `image_pad=151655`.
8. **Token selection for multi-vector (parity-critical).** The reference keeps **all non-pad tokens** in multi-vector mode — text: all tokens incl. `Query:`/`Passage:` prefix + special tokens; image: all attended tokens **incl. image-patch tokens**. (This differs from the dense path, which slices to the vision span. We must replicate the multi-vector rule exactly.)

### Consequence for design
vLLM gives us per-token **2048-dim** hidden states. We must apply the **retrieval-effective `multi_vector_projector` (128×2048 + bias)** ourselves, then per-token L2-normalize. The architectural question is **where** that projection runs — which is the three-variant study (§4).

---

## 3. Architecture Overview

```
                    ┌──────────────────────── Modal app (A10G 24GB) ────────────────────────┐
                    │                                                                        │
  probe inputs ───► │   vLLM pooling server  (jina-embeddings-v4-vllm-retrieval)             │
 (text + images)    │   `--runner pooling --pooler-config.task token_embed`                  │
                    │        │  /pooling → [n, 2048] per-token hidden states                 │
                    │        ▼                                                               │
                    │   projection site  ──►  [n, 128] multivectors  (per-token L2-norm)     │
                    │     (A: client | B: proxy | C: in-vLLM plugin)                         │
                    └────────────────────────────────────────────────────────────────────────┘
                                                  ▲
                                                  │ compare (element-wise)
  reference harness (local/GPU) ─────────────────┘
  HF transformers jina-embeddings-v4, retrieval adapter, return_multivector=True, fp32
                                  → ground-truth [n, 128] multivectors
```

A single Modal app hosts vLLM in pooling mode. The **reference harness** produces ground truth. A **shared `multivector` module** (token selection + projection + normalize + MaxSim) is imported by *both* the reference path and all variants, so any divergence is attributable to the backbone/serialization — never to our post-processing code. The **eval harness** computes element-wise parity and emits a report.

---

## 4. The Three Variants (all built, all measured)

The projection is mathematically identical across all three (`y = W·x + b`, then per-token L2-norm). They differ only in **where** it runs, which affects **serialization precision**, **projection dtype**, **latency**, **wire size**, and **complexity**.

| Variant | vLLM serving | Projection site | What it isolates / why it might differ |
|---|---|---|---|
| **A — Client-side** | stock `vllm serve --runner pooling --pooler-config.task token_embed` (returns `[n,2048]`) | Python client downloads projector once, applies locally | Serialization precision of the 2048-dim intermediate over HTTP; client simplicity; raw vectors cross the wire |
| **B — Server-side proxy** | same vLLM server, plus a thin FastAPI co-located on the same Modal GPU | proxy applies projector + normalize → returns `[n,128]` | Clean self-contained service (likely production shape); projection runs server-side (fp32 on CPU/GPU) |
| **C — In-vLLM plugin** | custom registered vLLM model/pooler with projector loaded in-engine | inside vLLM; native `/pooling` returns `[n,128]` | No intermediate serialization; projection in-engine (fp16); lowest latency; highest engineering cost / version-fragility |

**Expected outcome (to be confirmed):** A ≈ B ≈ C to within fp serialization/dtype epsilon; the dominant parity gap for *all three* is the vLLM fp16 backbone vs the fp32 reference, plus image-preprocessing drift — which affects all three equally.

---

## 5. Components (each one job, independently testable)

- **`reference/`** — HF `transformers` embedder. Loads `jinaai/jina-embeddings-v4`, retrieval adapter, fp32. Functions: `embed_texts(prompts, kind)` and `embed_images(images)` → ground-truth `[n,128]` multivectors **+ the token ids** used (for alignment). This is the only place the official Jina path runs.
- **`projector/`** — extracts and persists the **retrieval-effective** `multi_vector_projector` (base + retrieval-LoRA, merged → `128×2048` + bias) from the main repo. Saved to disk as the single source of truth consumed by A/B/C. Includes a unit test that the extracted matrix reproduces the reference projector output on random hidden states.
- **`multivector/`** — pure post-processing: token selection (per §2.8 rules), projection (apply saved weights), per-token L2-norm, and MaxSim scoring. Imported by `reference/` and all variants so the math is provably identical everywhere.
- **`vllm_core/`** — Modal deployment of the vLLM pooling server (GPU core for A & B): image build (pinned vllm), A10G, `@modal.web_server` over a `vllm serve` subprocess, HF cache volume, HF_TOKEN secret.
- **`variant_a/`** — client library: calls `/pooling`, applies `multivector` locally.
- **`variant_b/`** — FastAPI proxy (co-located on Modal) in front of `/pooling`; applies `multivector`; exposes a clean `/embeddings` returning `[n,128]` (+ token ids).
- **`variant_c/`** — custom vLLM model/pooler registration (out-of-tree plugin / `--trust-remote-code` modeling file) that loads the projector and returns `[n,128]` from native `/pooling`.
- **`data/probes/`** — probe inputs: a handful of text strings (short, long, multilingual, special chars — exercising prefix + special-token paths) and a few images (varying size/content). **No relevance labels.**
- **`eval/`** — runs reference + each variant on the probes, computes element-wise parity metrics (§6), emits a markdown + JSON report.

**Interface contract (shared across variants):** an embedding result is `{ "tokens": int, "dim": 128, "multivectors": float[n][128], "token_ids": int[n] }`. `token_ids` is mandatory so the eval harness can verify alignment before element-wise comparison.

---

## 6. Parity Methodology

### Deciding metric: direct output comparison (not retrieval ranking)
We expect near bit-identical outputs, so we compare embedding **tensors element-wise**. No curated corpus, no relevance labels, no NDCG/Recall/Kendall-τ.

For identical input, compare each variant's `[n,128]` tensor against the reference:
- **max absolute element difference** and **mean absolute difference** (expect ~fp16 epsilon),
- **per-token cosine similarity** distribution — report **min / mean** across tokens,
- **relative Frobenius norm** of the difference: `‖V − R‖_F / ‖R‖_F`.

**Pass thresholds (proposed, revisable after first measurement):**
- Offline (Stage 1): mean per-token cosine ≥ **0.999**, max abs diff < **1e-3**.
- Served (Stage 2): mean per-token cosine ≥ **0.99**.
These thresholds may be loosened with justification once we see the real fp16-backbone floor; the report states the achieved numbers regardless.

### Alignment precondition (also our image-drift detector)
Element-wise comparison requires **identical token count & ordering** between reference and vLLM.
- **Text:** guaranteed (same tokenizer) — assert equal `token_ids`.
- **Images:** token count depends on preprocessing (resize → patch count). The **first** image check is: *do both paths produce the same number of image tokens?* If not, that **is** the image-preprocessing drift risk (R2) surfacing — report it explicitly; do **not** silently align/truncate. If counts match, proceed to element-wise comparison.

### Two-stage structure
- **Stage 0 — ground truth:** HF reference at **fp32** (so backbone fp16 effects are attributable to vLLM, not the reference).
- **Stage 1 — offline:** vLLM offline `LLM.encode` with **exact prompts** (`TextPrompt` / `multi_modal_data`, bypassing chat-template ambiguity) + our projector vs reference. Isolates **projection correctness + backbone numerics**.
- **Stage 2 — served:** each of A/B/C over HTTP vs the Stage-1 offline result (and vs reference). Isolates **serialization + HTTP chat-template + image preprocessing**.

### Multimodal coverage
Embed both text and images and compare both numerically (the parity check itself proves both modalities work). One **optional** smoke test: a matched text↔image pair MaxSim-scores above an unmatched pair — sanity only, not a gate.

---

## 7. Tooling, Repo Layout, Environment

- **Language/tooling:** Python 3.12, `uv` for deps/venv. Pinned `vllm`, `transformers`, `torch`, `pillow`, `numpy`, `fastapi`, `modal`.
- **Compute:** Modal, **A10G 24GB**. HF_TOKEN as a Modal secret. HF model cache on a Modal volume.
- **Reference execution:** prefer GPU on Modal for the fp32 reference (3.8B fp32 ≈ 15GB — fits A10G 24GB); CPU fallback only if needed for spot checks.
- **Repo layout:**
  ```
  jina-hf/
    pyproject.toml            # uv project, pinned deps
    Makefile / justfile       # deploy, reference, parity, report targets
    src/jinav4_vllm/
      reference/              # HF transformers ground truth
      projector/             # extract retrieval-effective projector
      multivector/           # token selection + projection + norm + MaxSim
      vllm_core/             # Modal vLLM pooling server
      variant_a/  variant_b/  variant_c/
      eval/                  # parity metrics + report
    data/probes/             # probe text + images
    docs/superpowers/specs/  # this spec
    reports/                 # generated parity reports
  ```

---

## 8. Risks & Early Spikes (validate before full build)

| ID | Risk | Mitigation / spike |
|---|---|---|
| **R1** | Jina v4 multimodal over `/pooling` (HTTP) is undocumented — images may not pass via `messages`/`image_url` for this model | **Spike first.** If it fails, fall back to running the vLLM **offline engine inside** the variant B/C server process (still vLLM, still served) instead of the native `vllm serve` HTTP path |
| **R2** | Image preprocessing drift (vLLM transformers path vs `qwen_vl_utils`) → different image-token count/order, breaking element-wise compare | The alignment precondition (§6) detects it; if it triggers, pin/align the image processor or document the gap |
| **R3** | vLLM version churn around pooling (`token_embed` task, `pooling_type=ALL` regression #25165) | Pin a known-good vLLM version; verify offline `token_embed` returns `[n,2048]` before building servers |
| **R4** | Projector LoRA-merge correctness (must be retrieval-effective, not base) | `projector/` unit test: extracted matrix reproduces reference projector output on random hidden states within fp tolerance |
| **R5** | Variant C plugin feasibility / version-fragility | Time-box the spike; if infeasible on the pinned version, report C as "not viable on vLLM <ver>" and decide between A and B |

---

## 9. Sequencing

1. **Spikes:** R3 (offline `token_embed` shape) → R1 (multimodal `/pooling`) → R4 (projector extraction correctness).
2. **Foundations:** `projector/` + `reference/` + `multivector/`.
3. **Stage 1 offline parity** (proves the core math + backbone numerics).
4. **Variant B** (likely production shape) → served Stage-2 parity.
5. **Variant A** (cheap) → served Stage-2 parity.
6. **Variant C** (most effort) → served Stage-2 parity (or documented infeasibility).
7. **Comparative report + recommendation.**

---

## 10. Deliverables

- Reproducible `uv` repo with the components in §5.
- Deployed Modal endpoint(s) for the variants.
- A parity report (`reports/`) with element-wise numbers per variant per modality, the achieved fp16 floor, and a **production recommendation** among A/B/C.
