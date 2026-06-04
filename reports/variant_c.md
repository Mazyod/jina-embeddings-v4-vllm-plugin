# Variant C — in-vLLM pooler plugin (R5 time-boxed outcome, 2026-06-04)

## Decision: NOT implemented within the feasibility time-box; **Variant B recommended**. Feasible as a follow-up.

## Why C is hard on vLLM 0.22 (evidence)
`inspect_pooler` (app.py) shows vLLM 0.22 runs the **v1 engine**: the model executes in a separate
`EngineCore` subprocess. `LLMEngine` exposes no `model_executor`/model handle from the driver
process (`AttributeError: 'LLMEngine' object has no attribute 'model_executor'`; the engine only
exposes `apply_model`, `collective_rpc`, `engine_core`, ...). Consequently:

- You **cannot** monkeypatch the model's pooler from the serving process — it lives across a
  process boundary.
- The supported way to inject a custom pooler/projection is a **vLLM plugin**: an installed package
  exposing a `vllm.general_plugins` entry point that calls
  `ModelRegistry.register_model("JinaV4MultiVector", JinaV4MultiVector)` inside every worker
  process, plus serving with `--hf-overrides '{"architectures":["JinaV4MultiVector"]}'`.
- `JinaV4MultiVector` must subclass vLLM's Qwen2.5-VL pooling model and override the pooler to apply
  the `128×2048` projection + L2-norm. The exact base class and pooler `forward` signature are
  internal and were not reachable for introspection (model in subprocess), so getting this right is
  an iterative, version-fragile exercise.

## Why it isn't worth it for this study (evidence, not assumption)
The whole point of building all three sites was to test whether the **projection site** changes the
result (the "loss of information" doubt). It does not:

| site | text cos_mean | image | serialization of 2048-dim intermediate |
|---|---|---|---|
| offline (in-process) | 0.999 | aligned, cos_mean 0.992–0.997 | none |
| **variant_b** (server proxy, in-process engine) | **0.999** | **aligned, faithful** | none |
| **variant_a** (stock serve, raw 2048 over JSON → client projects) | **0.999** | chat-template token mismatch | full 2048-dim over the wire |

Variant A already pushes the raw 2048-dim hidden states across HTTP/JSON and **still** matches the
reference at cos_mean 0.999. So serialization precision — C's one distinguishing advantage — is
empirically a non-issue. C would reproduce the same 0.999 (identical projection math) while adding
the most implementation complexity and the most version risk.

## If native in-engine 128-dim output is later required (follow-up recipe)
1. Package a tiny plugin: `pyproject` entry point `vllm.general_plugins = {jina_v4_mv = "pkg:register"}`
   whose `register()` does `ModelRegistry.register_model("JinaV4MultiVector", JinaV4MultiVector)`.
2. `JinaV4MultiVector(<vLLM Qwen2.5-VL pooling base>)`: load `retrieval.npz` (W,b) in `__init__` (or
   from an env var path on a mounted volume); override the pooler so token-wise output =
   `normalize(hidden @ W.T + b)` → `[n,128]`.
3. Install the plugin into the vLLM image; `vllm serve jinaai/jina-embeddings-v4-vllm-retrieval
   --runner pooling --pooler-config.task token_embed --hf-overrides '{"architectures":["JinaV4MultiVector"]}'`.
4. Validate `/pooling` returns `[n,128]` and re-run the parity harness (expect ≈ variant_b).

Estimated effort: ~0.5–1 day including version-API spelunking; ongoing maintenance risk on vLLM
upgrades (the v1 pooler interface is internal/unstable).
