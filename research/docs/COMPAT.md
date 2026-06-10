# Compatibility & revalidation

The `JinaV4MultiVector` model class subclasses vLLM's `Qwen2_5_VLForConditionalGeneration` and calls
vLLM-internal pooler builders (`pooler_for_token_embed`, `default_pooling_type`, the multimodal
processor registry). These are **not** stable public APIs, so a vLLM upgrade can break the plugin
even when the model checkpoint is unchanged. **Pin the vLLM version** and revalidate on every bump.

## Tested matrix

| plugin | vLLM | transformers (reference only) | GPU | status |
|---|---|---|---|---|
| 0.1.0 | 0.22.0 | 4.57.6 (`<5`) | A10G | ✅ text + image parity (see `docs/VALIDATION.md`) |

`transformers` matters only for the HF *reference* path (Jina's `trust_remote_code` needs `<5`);
the vLLM serving path uses vLLM's native Qwen2.5-VL and is unaffected.

Where the version is pinned:
- `deploy/Dockerfile` — `ARG VLLM_TAG=v0.22.0` (the served image).
- `src/jinav4_vllm/modal_app/app.py` — the Modal `_vllm_base` image (pin at lock time).

## Revalidation checklist (run on a vLLM upgrade)

1. **Regenerate the API facts** the model class depends on (cheap, mostly CPU):
   ```bash
   make revalidate                                                  # recon_qwen25_api
   uv run modal run src/jinav4_vllm/modal_app/revalidate.py::recon_dump_sources
   uv run modal run src/jinav4_vllm/modal_app/revalidate.py::recon_vllm_pooling
   ```
   Confirm the symbols `model.py` imports still exist at the same paths: `pooler_for_token_embed`,
   `default_pooling_type`, `SupportsLateInteraction`, the `Qwen2_5_VL*` classes, and
   `MULTIMODAL_REGISTRY`. Update `model.py` if any moved.
2. **Live pooler shape** (GPU): the in-engine pooler still returns per-token vectors.
   ```bash
   uv run modal run src/jinav4_vllm/modal_app/revalidate.py::spike_offline_shape   # [n,2048] + ids
   uv run modal run src/jinav4_vllm/modal_app/revalidate.py::inspect_pooler        # model/pooler surface
   ```
3. **Projector still reproduces** the model's own head:
   ```bash
   uv run modal run src/jinav4_vllm/modal_app/revalidate.py::verify_projector      # max_abs_diff < 1e-3
   ```
4. **End-to-end parity** against the canonical reference:
   ```bash
   make e2e                                                         # extract -> reference -> offline -> parity
   ```
   Expect text cos_mean ≈ 0.999, image ≈ 0.992–0.997, all aligned.
5. **Served contract** smoke after redeploying:
   ```bash
   make serve
   make smoke URL=https://<your-deployment>                         # /pooling returns dim 128, L2-normalized
   ```
6. Bump the row in the matrix above and the pinned tags once green.

> Tip: `make package` builds the versioned plugin wheel; install it `--no-deps` into the official
> `vllm/vllm-openai:<TAG>` image so pip never re-resolves vLLM/torch.
