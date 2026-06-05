# Variant C — native vLLM OpenAI server (in-engine projection) — ✅ WORKING

**Status (2026-06-05): IMPLEMENTED and PASSING for text + image.** A stock `vllm serve` OpenAI
server returns final L2-normalized `[n,128]` multivectors directly from `/pooling`, for both text
and images. This is the "unify on the vLLM OpenAI image" route.

## Result

| probe | aligned | cos_min | cos_mean | pass |
|---|---|---|---|---|
| text_query_en_short | ✅ | 0.99732 | 0.99922 | PASS |
| text_passage_en_long | ✅ | 0.99552 | 0.99933 | PASS |
| text_query_ar | ✅ | 0.99732 | 0.99922 | PASS |
| text_query_ja | ✅ | 0.99732 | 0.99920 | PASS |
| text_passage_symbols | ✅ | 0.99658 | 0.99903 | PASS |
| image_cat | ✅ | 0.81420 | 0.99206 | PASS |
| image_chart | ✅ | 0.96162 | 0.99663 | PASS |

Identical parity to Variant B and the offline baseline (same projection math; the bf16 floor is the
only gap). `/pooling` returns `[n,128]`, row L2-norm = 1.0.

## How it works (3 small pieces on top of stock vLLM)

1. **A vLLM general plugin** (`src/jinav4_vllm/vllm_plugin/`, pip-installed into the image). It
   registers an out-of-tree architecture `JinaV4MultiVector` via the `vllm.general_plugins` entry
   point, so it loads in every process including the v1 `EngineCore` worker.
2. **The model class** `JinaV4MultiVectorModel` (`jina_v4_vllm_plugin/model.py`) — mirrors vLLM's
   in-tree ColQwen3 pattern: subclasses `Qwen2_5_VLForConditionalGeneration` + `SupportsLateInteraction`,
   `is_pooling_model = True`, `@default_pooling_type(tok_pooling_type="ALL")`, registers the
   Qwen2.5-VL multimodal processor. In `forward()` it runs the backbone, applies the
   `multi_vector_projector` (2048→128, bias, fp32), and L2-normalizes; `pooler_for_token_embed`
   gathers per-token. `load_weights` delegates the backbone to `super().load_weights` (inherited
   Qwen2.5-VL mapper) and injects the projector from the `.npz`.
3. **The projector weights** — not in the vLLM checkpoint; loaded at startup from
   `/artifacts/projector/retrieval.npz` (override via `JINA_MV_PROJECTOR`). For a fully self-contained
   image, bake the two tensors into the checkpoint instead (see follow-ups).
4. **A custom chat template** (`jina_image_chat_template.jinja`) so multimodal `/pooling` emits
   Jina's exact image prompt (`<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{text}<|im_end|>\n`)
   instead of the default Qwen template, which otherwise adds ~11 wrapper tokens and breaks alignment.

## Serve command

```bash
vllm serve jinaai/jina-embeddings-v4-vllm-retrieval \
  --runner pooling --pooler-config.task token_embed \
  --hf-overrides '{"architectures": ["JinaV4MultiVector"]}' \
  --chat-template jina_image_chat_template.jinja
# with the jina-v4-vllm-plugin package installed and retrieval.npz reachable
```

Client: `POST /pooling {"model": "...", "input": ["Query: ..."]}` for text → `data[i].data` is
`[n,128]`; for images, `messages` with `image_url` content. (Per-token output is the `/pooling`
endpoint; `/v1/embeddings` only returns one pooled vector per input.)

## Trade-offs vs B
- **Pro:** stock vLLM OpenAI image — unifies infra with other served LLMs; no separate FastAPI
  service; native `/pooling` contract.
- **Con:** carries a small out-of-tree model plugin whose base touches vLLM internals
  (`Qwen2_5_VLForConditionalGeneration`, pooler builders), so a vLLM upgrade may need the model class
  re-validated against the new pooling API (the recon functions in `app.py` regenerate the needed
  facts quickly).

## Follow-ups for productionizing C
1. **Bake the projector into a checkpoint** (add `multi_vector_projector.weight/bias` to a copy of the
   vLLM checkpoint + set `architectures` in its `config.json`) so `vllm serve <repo>` needs no env var
   and no `--hf-overrides` — fully drop-in.
2. Publish the plugin as a proper package (or upstream a `JinaV4` model to vLLM).
3. Pin the vLLM version; add a CI smoke test that `/pooling` returns dim 128 after upgrades.
