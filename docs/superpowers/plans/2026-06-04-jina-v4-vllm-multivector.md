# Jina v4 Multi-Vector Multimodal Embeddings on vLLM — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and prove a production-grade way to serve Jina Embeddings v4 multi-vector (128-dim/token, ColBERT-style) multimodal (text+image) embeddings on vLLM, comparing three projection-site architectures against the HF reference for numerical parity.

**Architecture:** vLLM (pooling mode, `token_embed`) on the adapter-merged checkpoint emits raw `[n,2048]` per-token hidden states; we apply the retrieval-effective `multi_vector_projector` (`128×2048` + bias, extracted from the main repo) + per-token L2-norm to get final multi-vectors. The projection is built in three sites — client-side (A), server-side proxy (B), in-vLLM plugin (C) — all measured against an HF `transformers` fp32 reference via element-wise parity over persisted `.npz` artifacts.

**Tech Stack:** Python 3.12, `uv`, NumPy (pure local logic), Modal (A10G GPU), vLLM (pooling server + offline engine), HF `transformers`/`peft`/`safetensors` (reference + projector), FastAPI (variant B), pytest.

**Spec:** `docs/superpowers/specs/2026-06-04-jina-v4-vllm-multivector-design.md`

---

## Conventions

- **Local dev env is lightweight** (NumPy + pytest + Modal client only). Torch/transformers/vLLM never install locally — they live in Modal images. This avoids the transformers⇄vLLM dependency conflict and needs no local GPU.
- **Two Modal images:** `ref_image` (torch + transformers + peft + safetensors) and `vllm_image` (vllm). They never mix in one process.
- **Artifacts** are `.npz` files on a Modal volume under `/artifacts/<source>/<probe_id>.npz`, each holding `mv` (float32 `[n,D]`, D∈{128,2048}) and `token_ids` (int64 `[n]`). Sources: `reference`, `offline`, `variant_a`, `variant_b`, `variant_c`.
- **Probe id** is a stable slug per probe input (e.g. `text_query_en_short`, `image_cat`).
- Every code step shows the full file or the full function. Commit after each green test.

---

## File Structure

```
jina-hf/
  pyproject.toml                         # uv project, light local deps
  Makefile                               # convenience targets
  src/jinav4_vllm/
    __init__.py
    multivector/
      __init__.py
      core.py                            # token select, projector apply, L2-norm, MaxSim  (NumPy, local)
    eval/
      __init__.py
      metrics.py                         # element-wise parity metrics  (NumPy, local)
      report.py                          # materialize artifacts + build report  (NumPy, local)
    projector/
      __init__.py
      merge.py                           # LoRA merge math  (NumPy, local)
      extract.py                         # extract retrieval-effective projector  (Modal, ref_image)
    common/
      __init__.py
      probes.py                          # probe definitions + prompt builders  (local)
      artifacts.py                       # npz read/write helpers  (NumPy, local)
    modal_app/
      __init__.py
      app.py                             # Modal app, images, volume, secret
      reference.py                       # HF reference harness  (Modal, ref_image)
      offline.py                         # vLLM offline engine harness  (Modal, vllm_image)
      serve_a.py                         # variant A: stock vllm serve  (Modal, vllm_image)
      serve_b.py                         # variant B: vllm + FastAPI proxy  (Modal, vllm_image)
      serve_c.py                         # variant C: in-vLLM plugin server  (Modal, vllm_image)
      client.py                          # HTTP client → artifacts  (Modal/local)
  data/probes/                           # probe images
  tests/
    test_multivector_core.py
    test_eval_metrics.py
    test_eval_report.py
    test_projector_merge.py
    test_probes.py
    test_artifacts.py
  reports/.gitkeep                       # generated parity reports land here
  artifacts/.gitkeep                     # local copies pulled from Modal volume
```

---

## Task 0: Scaffold the uv project

**Files:**
- Create: `pyproject.toml`, `Makefile`, `src/jinav4_vllm/__init__.py`, all package `__init__.py`, `reports/.gitkeep`, `artifacts/.gitkeep`, `tests/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "jinav4-vllm"
version = "0.1.0"
description = "Jina v4 multi-vector multimodal embeddings on vLLM — feasibility study"
requires-python = ">=3.12"
dependencies = [
    "numpy>=2.0",
    "pillow>=10.0",
    "modal>=0.64",
]

[dependency-groups]
dev = ["pytest>=8.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/jinav4_vllm"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 2: Create package skeleton**

```bash
mkdir -p src/jinav4_vllm/{multivector,eval,projector,common,modal_app} tests data/probes reports artifacts
touch src/jinav4_vllm/__init__.py \
      src/jinav4_vllm/multivector/__init__.py \
      src/jinav4_vllm/eval/__init__.py \
      src/jinav4_vllm/projector/__init__.py \
      src/jinav4_vllm/common/__init__.py \
      src/jinav4_vllm/modal_app/__init__.py \
      tests/__init__.py reports/.gitkeep artifacts/.gitkeep
```

- [ ] **Step 3: Create `Makefile`**

```makefile
.PHONY: install test deploy reference offline parity report
install:
	uv sync
test:
	uv run pytest -q
deploy:
	uv run modal deploy src/jinav4_vllm/modal_app/app.py
reference:
	uv run modal run src/jinav4_vllm/modal_app/reference.py
offline:
	uv run modal run src/jinav4_vllm/modal_app/offline.py
parity:
	uv run python -m jinav4_vllm.eval.report
```

- [ ] **Step 4: Sync and verify**

Run: `uv sync && uv run python -c "import jinav4_vllm; print('ok')"`
Expected: prints `ok` (after resolving numpy/pillow/modal).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "chore: scaffold uv project structure"
```

---

## Task 1: `multivector/core.py` — projection math (local TDD)

Pure NumPy. The single source of truth for token selection, projection, normalization, and MaxSim. Imported everywhere so the math is provably identical across reference and all variants.

**Files:**
- Create: `src/jinav4_vllm/multivector/core.py`
- Test: `tests/test_multivector_core.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_multivector_core.py
import numpy as np
import pytest
from jinav4_vllm.multivector.core import (
    apply_projector, l2_normalize, to_multivector, maxsim, select_tokens,
)

def test_apply_projector_matches_manual_linear():
    rng = np.random.default_rng(0)
    hidden = rng.standard_normal((5, 2048)).astype(np.float32)
    W = rng.standard_normal((128, 2048)).astype(np.float32)
    b = rng.standard_normal((128,)).astype(np.float32)
    out = apply_projector(hidden, W, b)
    assert out.shape == (5, 128)
    np.testing.assert_allclose(out, hidden @ W.T + b, rtol=1e-5, atol=1e-5)

def test_l2_normalize_unit_rows():
    rng = np.random.default_rng(1)
    mat = rng.standard_normal((4, 128)).astype(np.float32)
    out = l2_normalize(mat)
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, np.ones(4), rtol=1e-5, atol=1e-5)

def test_l2_normalize_zero_row_safe():
    mat = np.zeros((1, 8), dtype=np.float32)
    out = l2_normalize(mat)
    assert np.all(np.isfinite(out))           # no NaN from divide-by-zero
    assert np.allclose(out, 0.0)

def test_to_multivector_is_projector_then_norm():
    rng = np.random.default_rng(2)
    hidden = rng.standard_normal((3, 2048)).astype(np.float32)
    W = rng.standard_normal((128, 2048)).astype(np.float32)
    b = rng.standard_normal((128,)).astype(np.float32)
    mv = to_multivector(hidden, W, b)
    np.testing.assert_allclose(mv, l2_normalize(apply_projector(hidden, W, b)), rtol=1e-6, atol=1e-6)

def test_maxsim_identical_sequences_equals_token_count():
    rng = np.random.default_rng(3)
    q = l2_normalize(rng.standard_normal((6, 128)).astype(np.float32))
    # MaxSim of a normalized sequence against itself = sum of per-token max self-sim = n (each token matches itself at 1.0)
    score = maxsim(q, q)
    assert score == pytest.approx(6.0, abs=1e-4)

def test_maxsim_orthogonal_docs_lower_than_match():
    rng = np.random.default_rng(4)
    q = l2_normalize(rng.standard_normal((4, 128)).astype(np.float32))
    other = l2_normalize(rng.standard_normal((4, 128)).astype(np.float32))
    assert maxsim(q, q) > maxsim(q, other)

def test_select_tokens_drops_pad_with_mask():
    hidden = np.arange(12, dtype=np.float32).reshape(3, 4)
    ids = np.array([10, 11, 0], dtype=np.int64)
    mask = np.array([1, 1, 0], dtype=np.int64)
    h2, ids2 = select_tokens(hidden, ids, mask)
    assert h2.shape == (2, 4)
    np.testing.assert_array_equal(ids2, np.array([10, 11]))

def test_select_tokens_identity_without_mask():
    hidden = np.ones((3, 4), dtype=np.float32)
    ids = np.array([1, 2, 3], dtype=np.int64)
    h2, ids2 = select_tokens(hidden, ids, None)
    assert h2.shape == (3, 4)
    np.testing.assert_array_equal(ids2, ids)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_multivector_core.py -q`
Expected: FAIL — `ModuleNotFoundError: jinav4_vllm.multivector.core`.

- [ ] **Step 3: Write the implementation**

```python
# src/jinav4_vllm/multivector/core.py
"""Pure-NumPy multi-vector post-processing — the single source of projection math.

Per the spec (§2.8), multi-vector mode keeps all non-pad tokens (text: incl.
Query/Passage prefix + special tokens; image: incl. image-patch tokens). The
projection is the retrieval-effective `multi_vector_projector`: y = x @ W.T + b,
then per-token L2-normalize.
"""
from __future__ import annotations
import numpy as np


def select_tokens(
    hidden: np.ndarray, token_ids: np.ndarray, attention_mask: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Keep all non-pad tokens. With no mask (single unpadded sequence) this is identity."""
    if attention_mask is None:
        return hidden, token_ids
    keep = attention_mask.astype(bool)
    return hidden[keep], token_ids[keep]


def apply_projector(hidden: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Linear projection per token. hidden [n,2048], W [128,2048], b [128] -> [n,128]."""
    return hidden.astype(np.float32) @ W.astype(np.float32).T + b.astype(np.float32)


def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-row L2 normalize; zero rows stay zero (no NaN)."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, eps)


def to_multivector(hidden: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Full multi-vector: project then per-token L2-normalize."""
    return l2_normalize(apply_projector(hidden, W, b))


def maxsim(query_mv: np.ndarray, doc_mv: np.ndarray) -> float:
    """ColBERT late-interaction score: sum over query tokens of max over doc tokens of dot product."""
    sim = query_mv @ doc_mv.T              # [nq, nd]
    return float(sim.max(axis=1).sum())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_multivector_core.py -q`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add src/jinav4_vllm/multivector/core.py tests/test_multivector_core.py
git commit -m "feat: multivector core projection math (numpy, tested)"
```

---

## Task 2: `eval/metrics.py` — element-wise parity metrics (local TDD)

The deciding metric per the spec (§6): direct tensor comparison + the token-alignment precondition.

**Files:**
- Create: `src/jinav4_vllm/eval/metrics.py`
- Test: `tests/test_eval_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_metrics.py
import numpy as np
import pytest
from jinav4_vllm.eval.metrics import (
    assert_aligned, max_abs_diff, mean_abs_diff, per_token_cosine,
    relative_frobenius, compare, AlignmentError,
)

def test_assert_aligned_passes_on_equal_ids():
    ids = np.array([1, 2, 3])
    assert_aligned(ids, ids.copy())          # no raise

def test_assert_aligned_raises_on_length_mismatch():
    with pytest.raises(AlignmentError) as e:
        assert_aligned(np.array([1, 2, 3]), np.array([1, 2]))
    assert "length" in str(e.value).lower()

def test_assert_aligned_raises_on_order_mismatch():
    with pytest.raises(AlignmentError):
        assert_aligned(np.array([1, 2, 3]), np.array([1, 3, 2]))

def test_identical_tensors_have_zero_diff_and_unit_cosine():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 128)).astype(np.float32)
    assert max_abs_diff(A, A) == pytest.approx(0.0, abs=1e-7)
    assert mean_abs_diff(A, A) == pytest.approx(0.0, abs=1e-7)
    np.testing.assert_allclose(per_token_cosine(A, A), np.ones(5), atol=1e-6)
    assert relative_frobenius(A, A) == pytest.approx(0.0, abs=1e-7)

def test_max_abs_diff_picks_largest_element_gap():
    A = np.zeros((2, 2), dtype=np.float32)
    B = np.array([[0.0, 0.5], [0.0, -0.9]], dtype=np.float32)
    assert max_abs_diff(A, B) == pytest.approx(0.9)

def test_per_token_cosine_orthogonal_is_zero():
    A = np.array([[1.0, 0.0]], dtype=np.float32)
    B = np.array([[0.0, 1.0]], dtype=np.float32)
    assert per_token_cosine(A, B)[0] == pytest.approx(0.0, abs=1e-6)

def test_compare_bundles_expected_keys():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((4, 128)).astype(np.float32)
    B = A + 1e-4
    out = compare(A, B)
    for k in ("max_abs_diff", "mean_abs_diff", "cosine_min", "cosine_mean", "rel_frobenius", "n_tokens"):
        assert k in out
    assert out["n_tokens"] == 4
    assert out["cosine_min"] <= out["cosine_mean"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval_metrics.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

```python
# src/jinav4_vllm/eval/metrics.py
"""Element-wise parity metrics for comparing a variant's multivectors to the reference."""
from __future__ import annotations
import numpy as np


class AlignmentError(ValueError):
    """Raised when token sequences are not identical (count or order)."""


def assert_aligned(ids_a: np.ndarray, ids_b: np.ndarray) -> None:
    a = np.asarray(ids_a).ravel()
    b = np.asarray(ids_b).ravel()
    if a.shape[0] != b.shape[0]:
        raise AlignmentError(f"token length mismatch: {a.shape[0]} vs {b.shape[0]}")
    if not np.array_equal(a, b):
        raise AlignmentError("token id order mismatch")


def max_abs_diff(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.abs(A.astype(np.float64) - B.astype(np.float64)).max())


def mean_abs_diff(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.abs(A.astype(np.float64) - B.astype(np.float64)).mean())


def per_token_cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A.astype(np.float64); B = B.astype(np.float64)
    num = (A * B).sum(axis=1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    den = np.maximum(den, 1e-12)
    return num / den


def relative_frobenius(A: np.ndarray, B: np.ndarray) -> float:
    A = A.astype(np.float64); B = B.astype(np.float64)
    denom = np.linalg.norm(B)
    return float(np.linalg.norm(A - B) / max(denom, 1e-12))


def compare(variant: np.ndarray, reference: np.ndarray) -> dict:
    """All parity metrics for one probe. Assumes already token-aligned (same n)."""
    cos = per_token_cosine(variant, reference)
    return {
        "n_tokens": int(reference.shape[0]),
        "dim": int(reference.shape[1]),
        "max_abs_diff": max_abs_diff(variant, reference),
        "mean_abs_diff": mean_abs_diff(variant, reference),
        "cosine_min": float(cos.min()),
        "cosine_mean": float(cos.mean()),
        "rel_frobenius": relative_frobenius(variant, reference),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_eval_metrics.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add src/jinav4_vllm/eval/metrics.py tests/test_eval_metrics.py
git commit -m "feat: element-wise parity metrics + alignment precondition"
```

---

## Task 3: `projector/` — LoRA merge math (local TDD) + extraction (Modal)

Extract the **retrieval-effective** `multi_vector_projector` = base + retrieval-LoRA, merged. The merge math is tested locally; the real extraction + a correctness check run on Modal.

**Files:**
- Create: `src/jinav4_vllm/projector/merge.py`, `src/jinav4_vllm/projector/extract.py`
- Test: `tests/test_projector_merge.py`

- [ ] **Step 1: Write the failing test (merge math)**

```python
# tests/test_projector_merge.py
import numpy as np
import pytest
from jinav4_vllm.projector.merge import lora_delta, merge_linear

def test_lora_delta_shape_and_scaling():
    rng = np.random.default_rng(0)
    r, in_f, out_f = 4, 16, 8
    A = rng.standard_normal((r, in_f)).astype(np.float32)   # lora_A: [r, in]
    B = rng.standard_normal((out_f, r)).astype(np.float32)  # lora_B: [out, r]
    delta = lora_delta(A, B, alpha=8, r=4)
    assert delta.shape == (out_f, in_f)
    np.testing.assert_allclose(delta, (B @ A) * (8 / 4), rtol=1e-5, atol=1e-5)

def test_merge_linear_equals_base_plus_delta():
    rng = np.random.default_rng(1)
    base_W = rng.standard_normal((8, 16)).astype(np.float32)
    base_b = rng.standard_normal((8,)).astype(np.float32)
    A = rng.standard_normal((4, 16)).astype(np.float32)
    B = rng.standard_normal((8, 4)).astype(np.float32)
    W, b = merge_linear(base_W, base_b, A, B, alpha=8, r=4)
    np.testing.assert_allclose(W, base_W + (B @ A) * 2.0, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(b, base_b)   # LoRA does not touch bias

def test_merge_equivalent_forward():
    # The merged linear must equal applying base then adding the LoRA path on the SAME input.
    rng = np.random.default_rng(2)
    x = rng.standard_normal((5, 16)).astype(np.float32)
    base_W = rng.standard_normal((8, 16)).astype(np.float32)
    base_b = rng.standard_normal((8,)).astype(np.float32)
    A = rng.standard_normal((4, 16)).astype(np.float32)
    B = rng.standard_normal((8, 4)).astype(np.float32)
    W, b = merge_linear(base_W, base_b, A, B, alpha=8, r=4)
    merged_out = x @ W.T + b
    # PEFT forward: base(x) + scaling * (x @ A.T @ B.T)
    scaling = 8 / 4
    peft_out = (x @ base_W.T + base_b) + scaling * (x @ A.T) @ B.T
    np.testing.assert_allclose(merged_out, peft_out, rtol=1e-4, atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_projector_merge.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the merge implementation**

```python
# src/jinav4_vllm/projector/merge.py
"""LoRA merge math for a single Linear layer (the multi_vector_projector).

PEFT convention for a Linear with weight W [out,in]:
  lora_A: [r, in], lora_B: [out, r], scaling = alpha / r
  effective forward: W x + b + scaling * (B @ A) x
  => merged weight = W + scaling * (B @ A); bias unchanged (LoRA does not adapt bias).
"""
from __future__ import annotations
import numpy as np


def lora_delta(lora_A: np.ndarray, lora_B: np.ndarray, alpha: float, r: int) -> np.ndarray:
    scaling = alpha / r
    return (lora_B.astype(np.float32) @ lora_A.astype(np.float32)) * scaling


def merge_linear(
    base_W: np.ndarray, base_b: np.ndarray,
    lora_A: np.ndarray, lora_B: np.ndarray, alpha: float, r: int,
) -> tuple[np.ndarray, np.ndarray]:
    W = base_W.astype(np.float32) + lora_delta(lora_A, lora_B, alpha, r)
    return W, base_b.astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_projector_merge.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Write the extraction script (runs on Modal, ref_image)**

> Note: this imports `app`/`ref_image` from Task 5's `modal_app/app.py`. Implement Task 5's `app.py` first if executing strictly in order, or stub the import. Function body is complete here.

```python
# src/jinav4_vllm/projector/extract.py
"""Extract the retrieval-effective multi_vector_projector and save to the volume.

Run: uv run modal run src/jinav4_vllm/modal_app/app.py::extract_projector
Produces /artifacts/projector/retrieval.npz on the Modal volume: {W:[128,2048], b:[128]}.
"""
from __future__ import annotations


def extract_and_save(out_path: str) -> dict:
    """Executed inside the Modal ref_image container. Returns metadata dict."""
    import json, os
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from peft import PeftConfig
    from jinav4_vllm.projector.merge import merge_linear

    MAIN = "jinaai/jina-embeddings-v4"
    repo = snapshot_download(MAIN)

    # --- locate base projector weights across shards ---
    base_W = base_b = None
    for fn in os.listdir(repo):
        if fn.endswith(".safetensors"):
            with safe_open(os.path.join(repo, fn), framework="pt") as f:
                for k in f.keys():
                    if k.endswith("multi_vector_projector.weight"):
                        base_W = f.get_tensor(k).float().cpu().numpy()
                    elif k.endswith("multi_vector_projector.bias"):
                        base_b = f.get_tensor(k).float().cpu().numpy()
    assert base_W is not None and base_b is not None, "base projector not found"
    assert base_W.shape == (128, 2048), f"unexpected base_W shape {base_W.shape}"

    # --- locate retrieval adapter LoRA for the projector ---
    adapter_dir = os.path.join(repo, "adapters", "retrieval")
    if not os.path.isdir(adapter_dir):
        # fall back: adapters may be under a flat 'adapters' dir or named differently
        for cand in ("adapters/retrieval", "retrieval", "adapters"):
            p = os.path.join(repo, cand)
            if os.path.isdir(p) and any(x.startswith("adapter_") for x in os.listdir(p)):
                adapter_dir = p
                break
    cfg = PeftConfig.from_pretrained(adapter_dir)
    alpha, r = cfg.lora_alpha, cfg.r

    lora_A = lora_B = None
    for fn in os.listdir(adapter_dir):
        if fn.endswith(".safetensors"):
            with safe_open(os.path.join(adapter_dir, fn), framework="pt") as f:
                for k in f.keys():
                    if "multi_vector_projector" in k and "lora_A" in k:
                        lora_A = f.get_tensor(k).float().cpu().numpy()
                    elif "multi_vector_projector" in k and "lora_B" in k:
                        lora_B = f.get_tensor(k).float().cpu().numpy()

    if lora_A is not None and lora_B is not None:
        W, b = merge_linear(base_W, base_b, lora_A, lora_B, alpha=alpha, r=r)
        merged = True
    else:
        # adapter does not LoRA the projector (modules_to_save case) — verified in Step 7
        W, b = base_W.astype("float32"), base_b.astype("float32")
        merged = False

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, W=W.astype("float32"), b=b.astype("float32"))
    return {"merged_lora": merged, "alpha": float(alpha), "r": int(r),
            "W_shape": list(W.shape), "b_shape": list(b.shape)}
```

- [ ] **Step 6: Run extraction on Modal**

Run: `uv run modal run src/jinav4_vllm/modal_app/app.py::extract_projector`
Expected: prints metadata e.g. `{'merged_lora': True, 'alpha': 32.0, 'r': 32, 'W_shape': [128, 2048], 'b_shape': [128]}` and writes `/artifacts/projector/retrieval.npz`.
**Decision branch:** if `merged_lora` is False, the adapter stores the projector via `modules_to_save` rather than LoRA — in that case adjust `extract_and_save` to read the full projector tensor directly from the adapter dir (key contains `multi_vector_projector` + `.weight`/`.bias` without `lora_`), then re-run. Document which path was taken in `reports/projector_extraction.md`.

- [ ] **Step 7: Correctness check on Modal (R4 mitigation)**

Add this verifier to `app.py` (full code in Task 5) and run it: load the full `jinaai/jina-embeddings-v4` model with the retrieval adapter active, feed random `[16,2048]` hidden states through the model's own `multi_vector_projector`, and compare to our extracted `W,b`.

Run: `uv run modal run src/jinav4_vllm/modal_app/app.py::verify_projector`
Expected: prints `max_abs_diff < 1e-3` and `PASS`. If it fails, the merge convention is wrong — inspect adapter key names and `task_label` handling before proceeding.

- [ ] **Step 8: Commit**

```bash
git add src/jinav4_vllm/projector/ tests/test_projector_merge.py
git commit -m "feat: retrieval-effective projector extraction + LoRA merge (tested)"
```

---

## Task 4: `common/` — probes, prompts, artifacts (local TDD)

**Files:**
- Create: `src/jinav4_vllm/common/probes.py`, `src/jinav4_vllm/common/artifacts.py`
- Test: `tests/test_probes.py`, `tests/test_artifacts.py`
- Create: a few small images in `data/probes/`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_probes.py
from jinav4_vllm.common.probes import TEXT_PROBES, IMAGE_PROBES, build_text_prompt, build_image_prompt

def test_text_probes_cover_required_cases():
    kinds = {p.kind for p in TEXT_PROBES}
    assert {"query", "passage"} <= kinds
    langs = {p.lang for p in TEXT_PROBES}
    assert len(langs) >= 2                      # multilingual coverage
    assert any(len(p.text) > 200 for p in TEXT_PROBES)   # a long one

def test_build_text_prompt_prefixes():
    assert build_text_prompt("hello", "query") == "Query: hello"
    assert build_text_prompt("world", "passage") == "Passage: world"

def test_build_image_prompt_is_exact_template():
    assert build_image_prompt() == (
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        "Describe the image.<|im_end|>\n"
    )

def test_image_probes_exist_and_have_paths():
    assert len(IMAGE_PROBES) >= 2
    for p in IMAGE_PROBES:
        assert p.path.endswith((".png", ".jpg", ".jpeg"))
```

```python
# tests/test_artifacts.py
import numpy as np
from jinav4_vllm.common.artifacts import save_artifact, load_artifact

def test_roundtrip(tmp_path):
    mv = np.random.default_rng(0).standard_normal((5, 128)).astype(np.float32)
    ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    p = tmp_path / "probe.npz"
    save_artifact(str(p), mv, ids)
    mv2, ids2 = load_artifact(str(p))
    np.testing.assert_array_equal(mv, mv2)
    np.testing.assert_array_equal(ids, ids2)
    assert mv2.dtype == np.float32 and ids2.dtype == np.int64
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_probes.py tests/test_artifacts.py -q`
Expected: FAIL — modules not found.

- [ ] **Step 3: Implement `artifacts.py`**

```python
# src/jinav4_vllm/common/artifacts.py
"""Persist/read multivector artifacts as .npz: mv [n,D] float32, token_ids [n] int64."""
from __future__ import annotations
import numpy as np


def save_artifact(path: str, mv: np.ndarray, token_ids: np.ndarray) -> None:
    np.savez(path, mv=mv.astype(np.float32), token_ids=np.asarray(token_ids, dtype=np.int64))


def load_artifact(path: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    return d["mv"].astype(np.float32), d["token_ids"].astype(np.int64)
```

- [ ] **Step 4: Implement `probes.py`**

```python
# src/jinav4_vllm/common/probes.py
"""Probe inputs (not a benchmark corpus): enough to exercise text + image paths."""
from __future__ import annotations
from dataclasses import dataclass

IMAGE_TEMPLATE = (
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "Describe the image.<|im_end|>\n"
)


@dataclass(frozen=True)
class TextProbe:
    id: str
    text: str
    kind: str          # "query" | "passage"
    lang: str


@dataclass(frozen=True)
class ImageProbe:
    id: str
    path: str


def build_text_prompt(text: str, kind: str) -> str:
    prefix = {"query": "Query", "passage": "Passage"}[kind]
    return f"{prefix}: {text}"


def build_image_prompt() -> str:
    return IMAGE_TEMPLATE


TEXT_PROBES: list[TextProbe] = [
    TextProbe("text_query_en_short", "Overview of climate change impacts on coastal cities", "query", "en"),
    TextProbe("text_passage_en_long",
              "The impacts of climate change on coastal cities are significant and far-reaching. "
              "Rising sea levels threaten infrastructure, increase flooding frequency, and force "
              "costly adaptation measures across transport, housing, and water systems. Storm "
              "surges compound the damage, while saltwater intrusion degrades freshwater supplies "
              "and agricultural land in low-lying delta regions worldwide.", "passage", "en"),
    TextProbe("text_query_ar", "تأثير تغير المناخ على المدن الساحلية", "query", "ar"),
    TextProbe("text_query_ja", "浜辺に沈む美しい夕日", "query", "ja"),
    TextProbe("text_passage_symbols", "Δ-encoding: cost ≈ $1,234.56 (≤2% error) — see §3.1 & Fig. 2.", "passage", "en"),
]

IMAGE_PROBES: list[ImageProbe] = [
    ImageProbe("image_cat", "data/probes/cat.png"),
    ImageProbe("image_chart", "data/probes/chart.png"),
]
```

- [ ] **Step 5: Create probe images**

```bash
uv run python - <<'PY'
from PIL import Image, ImageDraw
img = Image.new("RGB", (224, 224), (180, 200, 230))
d = ImageDraw.Draw(img); d.ellipse((60, 60, 164, 164), fill=(90, 90, 90)); d.text((70, 110), "cat", fill="white")
img.save("data/probes/cat.png")
img2 = Image.new("RGB", (320, 240), "white")
d2 = ImageDraw.Draw(img2)
for i, h in enumerate([40, 90, 60, 130, 100]):
    d2.rectangle((30 + i * 55, 200 - h, 70 + i * 55, 200), fill=(40, 120, 200))
d2.text((110, 10), "bar chart", fill="black")
img2.save("data/probes/chart.png")
print("wrote probe images")
PY
```

- [ ] **Step 6: Run to verify pass**

Run: `uv run pytest tests/test_probes.py tests/test_artifacts.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/jinav4_vllm/common/ tests/test_probes.py tests/test_artifacts.py data/probes/
git commit -m "feat: probe inputs, prompt builders, artifact io (tested)"
```

---

## Task 5: Modal app foundation + Spike R3 (offline `token_embed` shape)

**Files:**
- Create: `src/jinav4_vllm/modal_app/app.py`

- [ ] **Step 1: Write the Modal app (images, volume, secret, projector jobs, R3 spike)**

```python
# src/jinav4_vllm/modal_app/app.py
"""Modal app: images, volume, secret, and the foundational GPU jobs."""
from __future__ import annotations
import modal

app = modal.App("jinav4-vllm")

# Persistent volumes for HF cache and our artifacts.
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
artifacts = modal.Volume.from_name("jinav4-artifacts", create_if_missing=True)
CACHE = "/root/.cache/huggingface"
ART = "/artifacts"


def _with_local(img):
    """Attach project source + probe images so containers can import jinav4_vllm and read probes.
    Uses the current Modal idiom (add_local_dir). If the installed Modal version lacks it,
    fall back to `mounts=[modal.Mount.from_local_dir(...)]` on each @app.function."""
    return (
        img.add_local_dir("src/jinav4_vllm", remote_path="/root/jinav4_vllm")
           .add_local_dir("data/probes", remote_path="/root/data/probes")
    )


# Reference image: transformers stack.
ref_image = _with_local(
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch", "transformers>=4.52", "peft>=0.11", "safetensors",
        "huggingface_hub", "pillow", "numpy>=2.0", "accelerate",
    )
    .env({"HF_HOME": CACHE})
)

# vLLM image: pin at lock time (see Step 3). Kept separate from transformers stack.
vllm_image = _with_local(
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm", "pillow", "numpy>=2.0", "huggingface_hub")
    .env({"HF_HOME": CACHE})
)

HF_SECRET = modal.Secret.from_name("huggingface-secret")   # holds HF_TOKEN
GPU = "A10G"
COMMON = dict(volumes={CACHE: hf_cache, ART: artifacts}, secrets=[HF_SECRET])


@app.function(image=ref_image, timeout=1800, **COMMON)
def extract_projector():
    import sys; sys.path.insert(0, "/root")
    from jinav4_vllm.projector.extract import extract_and_save
    meta = extract_and_save(f"{ART}/projector/retrieval.npz")
    artifacts.commit()
    print(meta)
    return meta


@app.function(image=ref_image, gpu=GPU, timeout=1800, **COMMON)
def verify_projector():
    """R4: our extracted W,b must reproduce the model's own retrieval projector."""
    import sys; sys.path.insert(0, "/root")
    import numpy as np, torch
    from transformers import AutoModel
    proj = np.load(f"{ART}/projector/retrieval.npz")
    W, b = proj["W"], proj["b"]
    model = AutoModel.from_pretrained("jinaai/jina-embeddings-v4", trust_remote_code=True,
                                      torch_dtype=torch.float32).eval()
    # Activate retrieval adapter if the model exposes adapter switching.
    for setter in ("set_adapter", "set_task"):
        if hasattr(model, setter):
            try: getattr(model, setter)("retrieval")
            except Exception: pass
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.standard_normal((16, 2048)), dtype=torch.float32)
    with torch.no_grad():
        ref = model.multi_vector_projector(x).cpu().numpy()
    ours = x.numpy() @ W.T + b
    mad = float(np.abs(ref - ours).max())
    print(f"max_abs_diff={mad:.6e}", "PASS" if mad < 1e-3 else "FAIL")
    assert mad < 1e-3, "extracted projector does not match model projector"
    return mad


@app.function(image=vllm_image, gpu=GPU, timeout=1800, **COMMON)
def spike_r3_offline_shape():
    """R3: confirm offline pooling returns [n, 2048] per-token hidden states + token ids."""
    from vllm import LLM
    from vllm.config import PoolerConfig
    llm = LLM(model="jinaai/jina-embeddings-v4-vllm-retrieval",
              runner="pooling", pooler_config=PoolerConfig(task="token_embed"),
              max_model_len=1024, gpu_memory_utilization=0.8)
    out = llm.encode(["Query: hello world"])[0]
    data = out.outputs.data
    shape = tuple(data.shape)
    ntok = len(out.prompt_token_ids)
    print(f"data.shape={shape} n_prompt_tokens={ntok}")
    assert shape[1] == 2048, f"expected hidden dim 2048, got {shape[1]}"
    assert shape[0] == ntok, "row count must equal prompt token count"
    return {"shape": shape, "n_tokens": ntok}
```

- [ ] **Step 2: Create the HF secret (one-time, interactive)**

If not already present, the operator runs (paste into the session with `!` prefix so output is captured):
`! uv run modal secret create huggingface-secret HF_TOKEN=<your_hf_token>`
Expected: `Created secret huggingface-secret`.

- [ ] **Step 3: Pin the vLLM version and confirm the API (R3)**

Run: `uv run modal run src/jinav4_vllm/modal_app/app.py::spike_r3_offline_shape`
Expected: prints `data.shape=(N, 2048) n_prompt_tokens=N` and returns without assertion error.
**If `PoolerConfig(task="token_embed")` raises** (API drift), try the documented alternatives in order and keep the one that works, recording it in `reports/vllm_version.md`:
  1. server-equivalent offline: `LLM(..., runner="pooling", override_pooler_config=PoolerConfig(pooling_type="ALL", normalize=False))`
  2. `task="embed"` legacy form.
Then pin that exact vllm version: edit `vllm_image` to `.uv_pip_install("vllm==<resolved_version>", ...)` (find it via a one-off `print(vllm.__version__)` in the function).

- [ ] **Step 4: Commit**

```bash
git add src/jinav4_vllm/modal_app/app.py reports/
git commit -m "feat: modal app foundation, projector jobs, R3 offline-shape spike"
```

---

## Task 6: Reference harness (Modal, ref_image) — text ground truth

**Files:**
- Create: `src/jinav4_vllm/modal_app/reference.py`

- [ ] **Step 1: Write the reference harness**

```python
# src/jinav4_vllm/modal_app/reference.py
"""HF transformers ground truth. Writes /artifacts/reference/<id>.npz (mv [n,128], token_ids)."""
from __future__ import annotations
import modal
from jinav4_vllm.modal_app.app import app, ref_image, GPU, COMMON, ART, artifacts


@app.function(image=ref_image, gpu=GPU, timeout=2400, **COMMON)
def reference_text():
    import sys; sys.path.insert(0, "/root")
    import numpy as np, torch
    from transformers import AutoModel, AutoProcessor
    from jinav4_vllm.common.probes import TEXT_PROBES, build_text_prompt
    from jinav4_vllm.common.artifacts import save_artifact
    import os

    model = AutoModel.from_pretrained("jinaai/jina-embeddings-v4", trust_remote_code=True,
                                      torch_dtype=torch.float32).eval()
    processor = AutoProcessor.from_pretrained("jinaai/jina-embeddings-v4", trust_remote_code=True)
    for setter in ("set_adapter", "set_task"):
        if hasattr(model, setter):
            try: getattr(model, setter)("retrieval")
            except Exception: pass

    os.makedirs(f"{ART}/reference", exist_ok=True)
    results = {}
    for p in TEXT_PROBES:
        # Ground-truth values from the official high-level API (authoritative).
        mv = model.encode_text(texts=[p.text], task="retrieval",
                               prompt_name=p.kind, return_multivector=True)[0]
        mv = np.asarray(mv.detach().cpu().numpy() if hasattr(mv, "detach") else mv, dtype=np.float32)
        # Token ids from the identical prompt string (for alignment vs vLLM).
        prompt = build_text_prompt(p.text, p.kind)
        ids = np.asarray(processor.tokenizer(prompt, add_special_tokens=True)["input_ids"], dtype=np.int64)
        # Sanity: high-level multivector row count must equal our tokenization length.
        assert mv.shape[0] == ids.shape[0], (
            f"{p.id}: mv rows {mv.shape[0]} != token len {ids.shape[0]} "
            "(prompt/tokenization mismatch — fix prompt_name/template before trusting parity)")
        save_artifact(f"{ART}/reference/{p.id}.npz", mv, ids)
        results[p.id] = list(mv.shape)
    artifacts.commit()
    print(results)
    return results
```

- [ ] **Step 2: Run the reference harness**

Run: `uv run modal run src/jinav4_vllm/modal_app/reference.py::reference_text`
Expected: prints a dict of `{probe_id: [n, 128]}` and writes `/artifacts/reference/*.npz`.
**If the `assert mv.shape[0] == ids.shape[0]` fails:** the high-level prompt differs from `build_text_prompt` (e.g. extra special tokens). Adjust `build_text_prompt`/`add_special_tokens` so the tokenization matches the model's internal prompt exactly, then re-run. Record the resolved prompt format in `reports/prompt_format.md`.

- [ ] **Step 3: Commit**

```bash
git add src/jinav4_vllm/modal_app/reference.py reports/
git commit -m "feat: HF reference harness (text ground truth)"
```

---

## Task 7: Offline vLLM harness (Modal, vllm_image) — text raw hidden states

**Files:**
- Create: `src/jinav4_vllm/modal_app/offline.py`

- [ ] **Step 1: Write the offline harness**

```python
# src/jinav4_vllm/modal_app/offline.py
"""vLLM offline engine. Writes /artifacts/offline/<id>.npz (raw mv [n,2048], token_ids)."""
from __future__ import annotations
import modal
from jinav4_vllm.modal_app.app import app, vllm_image, GPU, COMMON, ART, artifacts


@app.function(image=vllm_image, gpu=GPU, timeout=2400, **COMMON)
def offline_text():
    import sys; sys.path.insert(0, "/root")
    import os, numpy as np
    from vllm import LLM
    from vllm.config import PoolerConfig
    from vllm.inputs.data import TextPrompt
    from jinav4_vllm.common.probes import TEXT_PROBES, build_text_prompt
    from jinav4_vllm.common.artifacts import save_artifact

    llm = LLM(model="jinaai/jina-embeddings-v4-vllm-retrieval",
              runner="pooling", pooler_config=PoolerConfig(task="token_embed"),
              max_model_len=2048, gpu_memory_utilization=0.85)

    os.makedirs(f"{ART}/offline", exist_ok=True)
    prompts = [TextPrompt(prompt=build_text_prompt(p.text, p.kind)) for p in TEXT_PROBES]
    outputs = llm.encode(prompts)
    results = {}
    for p, out in zip(TEXT_PROBES, outputs):
        hidden = out.outputs.data
        hidden = np.asarray(hidden.detach().cpu().numpy() if hasattr(hidden, "detach") else hidden,
                            dtype=np.float32)            # [n, 2048]
        ids = np.asarray(out.prompt_token_ids, dtype=np.int64)
        save_artifact(f"{ART}/offline/{p.id}.npz", hidden, ids)
        results[p.id] = list(hidden.shape)
    artifacts.commit()
    print(results)
    return results
```

- [ ] **Step 2: Run the offline harness**

Run: `uv run modal run src/jinav4_vllm/modal_app/offline.py::offline_text`
Expected: prints `{probe_id: [n, 2048]}` and writes `/artifacts/offline/*.npz`.

- [ ] **Step 3: Commit**

```bash
git add src/jinav4_vllm/modal_app/offline.py
git commit -m "feat: vLLM offline harness (text raw hidden states)"
```

---

## Task 8: `eval/report.py` + Stage 1 offline parity — TEXT

Materialize artifacts (project `[n,2048]`→`[n,128]` when needed), enforce alignment, compute metrics, write report. Tested locally with synthetic artifacts; then run for real over the pulled Modal artifacts.

**Files:**
- Create: `src/jinav4_vllm/eval/report.py`
- Test: `tests/test_eval_report.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_report.py
import numpy as np
from jinav4_vllm.eval.report import materialize, run_comparison
from jinav4_vllm.common.artifacts import save_artifact

def test_materialize_projects_2048_to_128(tmp_path):
    rng = np.random.default_rng(0)
    hidden = rng.standard_normal((4, 2048)).astype(np.float32)
    ids = np.arange(4, dtype=np.int64)
    p = tmp_path / "off.npz"; save_artifact(str(p), hidden, ids)
    W = rng.standard_normal((128, 2048)).astype(np.float32)
    b = rng.standard_normal((128,)).astype(np.float32)
    mv, got_ids = materialize(str(p), W, b)
    assert mv.shape == (4, 128)
    norms = np.linalg.norm(mv, axis=1)
    np.testing.assert_allclose(norms, np.ones(4), atol=1e-5)   # normalized
    np.testing.assert_array_equal(got_ids, ids)

def test_materialize_passthrough_128(tmp_path):
    mv = np.random.default_rng(1).standard_normal((3, 128)).astype(np.float32)
    ids = np.arange(3, dtype=np.int64)
    p = tmp_path / "ref.npz"; save_artifact(str(p), mv, ids)
    out, _ = materialize(str(p), None, None)
    np.testing.assert_array_equal(out, mv)                     # already 128: untouched

def test_run_comparison_perfect_match(tmp_path):
    rng = np.random.default_rng(2)
    hidden = rng.standard_normal((5, 2048)).astype(np.float32)
    ids = np.arange(5, dtype=np.int64)
    W = rng.standard_normal((128, 2048)).astype(np.float32)
    b = rng.standard_normal((128,)).astype(np.float32)
    from jinav4_vllm.multivector.core import to_multivector
    ref_mv = to_multivector(hidden, W, b)
    ref = tmp_path / "reference"; off = tmp_path / "offline"; ref.mkdir(); off.mkdir()
    save_artifact(str(ref / "x.npz"), ref_mv, ids)
    save_artifact(str(off / "x.npz"), hidden, ids)
    rows = run_comparison(str(ref), {"offline": str(off)}, W, b, ["x"])
    r = rows[0]
    assert r["source"] == "offline"
    assert r["cosine_min"] > 0.9999
    assert r["max_abs_diff"] < 1e-4
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_eval_report.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `report.py`**

```python
# src/jinav4_vllm/eval/report.py
"""Materialize artifacts to [n,128] and compare each variant to the reference."""
from __future__ import annotations
import os, json
import numpy as np
from jinav4_vllm.common.artifacts import load_artifact
from jinav4_vllm.multivector.core import to_multivector
from jinav4_vllm.eval.metrics import assert_aligned, compare, AlignmentError


def materialize(path: str, W: np.ndarray | None, b: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Load an artifact and return final [n,128] multivectors. Projects if stored at 2048."""
    mv, ids = load_artifact(path)
    if mv.shape[1] == 2048:
        assert W is not None and b is not None, "need projector to materialize 2048-dim artifact"
        mv = to_multivector(mv, W, b)
    return mv, ids


def run_comparison(ref_dir: str, source_dirs: dict[str, str],
                   W: np.ndarray | None, b: np.ndarray | None,
                   probe_ids: list[str]) -> list[dict]:
    rows: list[dict] = []
    for pid in probe_ids:
        ref_path = os.path.join(ref_dir, f"{pid}.npz")
        if not os.path.exists(ref_path):
            continue
        ref_mv, ref_ids = materialize(ref_path, W, b)
        for source, d in source_dirs.items():
            ap = os.path.join(d, f"{pid}.npz")
            if not os.path.exists(ap):
                continue
            mv, ids = materialize(ap, W, b)
            row = {"probe_id": pid, "source": source}
            try:
                assert_aligned(ids, ref_ids)
                row["aligned"] = True
                row.update(compare(mv, ref_mv))
            except AlignmentError as e:
                row["aligned"] = False
                row["error"] = str(e)
                row["ref_tokens"] = int(ref_ids.shape[0])
                row["var_tokens"] = int(ids.shape[0])
            rows.append(row)
    return rows


def render_markdown(rows: list[dict], thresholds: dict) -> str:
    lines = ["# Parity Report", "", "| probe | source | aligned | n | max_abs | mean_abs | cos_min | cos_mean | relF | pass |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if not r.get("aligned", False):
            lines.append(f"| {r['probe_id']} | {r['source']} | ❌ | - | - | - | - | - | - | "
                         f"MISALIGNED ({r.get('ref_tokens')}≠{r.get('var_tokens')}) |")
            continue
        ok = (r["cosine_mean"] >= thresholds["cosine_mean"] and r["max_abs_diff"] <= thresholds["max_abs_diff"])
        lines.append(f"| {r['probe_id']} | {r['source']} | ✅ | {r['n_tokens']} | "
                     f"{r['max_abs_diff']:.2e} | {r['mean_abs_diff']:.2e} | {r['cosine_min']:.5f} | "
                     f"{r['cosine_mean']:.5f} | {r['rel_frobenius']:.2e} | {'PASS' if ok else 'FAIL'} |")
    return "\n".join(lines) + "\n"


def main():
    """CLI: compares artifacts/ pulled from the Modal volume. Stage 1 = offline vs reference."""
    proj = np.load("artifacts/projector/retrieval.npz")
    W, b = proj["W"], proj["b"]
    from jinav4_vllm.common.probes import TEXT_PROBES, IMAGE_PROBES
    probe_ids = [p.id for p in TEXT_PROBES] + [p.id for p in IMAGE_PROBES]
    sources = {name: f"artifacts/{name}" for name in ("offline", "variant_a", "variant_b", "variant_c")
               if os.path.isdir(f"artifacts/{name}")}
    rows = run_comparison("artifacts/reference", sources, W, b, probe_ids)
    thresholds = {"cosine_mean": 0.99, "max_abs_diff": 1e-3}
    md = render_markdown(rows, thresholds)
    os.makedirs("reports", exist_ok=True)
    with open("reports/parity.md", "w") as f: f.write(md)
    with open("reports/parity.json", "w") as f: json.dump(rows, f, indent=2)
    print(md)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_eval_report.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Pull artifacts and run real Stage-1 text parity**

```bash
uv run modal volume get jinav4-artifacts /projector artifacts/ --force
uv run modal volume get jinav4-artifacts /reference artifacts/ --force
uv run modal volume get jinav4-artifacts /offline   artifacts/ --force
uv run python -m jinav4_vllm.eval.report
```
Expected: prints the parity table; offline-vs-reference text rows show `cos_mean ≥ 0.999`, `max_abs ~1e-3` or lower, `PASS`.
**If parity is poor (cos_mean < 0.99):** isolate — (a) re-run `verify_projector` (projector wrong?), (b) check the reference dtype is fp32 and vLLM dtype, (c) confirm token_ids aligned (table shows aligned ✅). Record findings in `reports/stage1_text.md`.

- [ ] **Step 6: Commit**

```bash
git add src/jinav4_vllm/eval/report.py tests/test_eval_report.py reports/
git commit -m "feat: parity report + Stage-1 offline text parity"
```

---

## Task 9: Image path — reference + offline + Spike R2 alignment + Stage 1 image parity

**Files:**
- Modify: `src/jinav4_vllm/modal_app/reference.py` (add `reference_image`)
- Modify: `src/jinav4_vllm/modal_app/offline.py` (add `offline_image`)

- [ ] **Step 1: Add `reference_image` to `reference.py`**

```python
@app.function(image=ref_image, gpu=GPU, timeout=2400, **COMMON)
def reference_image():
    import sys; sys.path.insert(0, "/root")
    import os, numpy as np, torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    from jinav4_vllm.common.probes import IMAGE_PROBES, build_image_prompt
    from jinav4_vllm.common.artifacts import save_artifact

    model = AutoModel.from_pretrained("jinaai/jina-embeddings-v4", trust_remote_code=True,
                                      torch_dtype=torch.float32).eval()
    processor = AutoProcessor.from_pretrained("jinaai/jina-embeddings-v4", trust_remote_code=True)
    for setter in ("set_adapter", "set_task"):
        if hasattr(model, setter):
            try: getattr(model, setter)("retrieval")
            except Exception: pass

    os.makedirs(f"{ART}/reference", exist_ok=True)
    results = {}
    for p in IMAGE_PROBES:
        img = Image.open(f"/root/data/probes/{os.path.basename(p.path)}").convert("RGB")
        mv = model.encode_image(images=[img], task="retrieval", return_multivector=True)[0]
        mv = np.asarray(mv.detach().cpu().numpy() if hasattr(mv, "detach") else mv, dtype=np.float32)
        # Token ids via the identical prompt + processor (vision tokens expand here).
        proc = processor(text=[build_image_prompt()], images=[img], return_tensors="pt")
        ids = np.asarray(proc["input_ids"][0].cpu().numpy(), dtype=np.int64)
        assert mv.shape[0] == ids.shape[0], (
            f"{p.id}: image mv rows {mv.shape[0]} != token len {ids.shape[0]}")
        save_artifact(f"{ART}/reference/{p.id}.npz", mv, ids)
        results[p.id] = list(mv.shape)
    artifacts.commit()
    print(results)
    return results
```

> Probe images are already available at `/root/data/probes/` because `app.py`'s `_with_local()` attaches `data/probes` to both images. No mount change needed.

- [ ] **Step 2: Add `offline_image` to `offline.py`**

```python
@app.function(image=vllm_image, gpu=GPU, timeout=2400, **COMMON)
def offline_image():
    import sys; sys.path.insert(0, "/root")
    import os, numpy as np
    from PIL import Image
    from vllm import LLM
    from vllm.config import PoolerConfig
    from vllm.inputs.data import TextPrompt
    from jinav4_vllm.common.probes import IMAGE_PROBES, build_image_prompt
    from jinav4_vllm.common.artifacts import save_artifact

    llm = LLM(model="jinaai/jina-embeddings-v4-vllm-retrieval",
              runner="pooling", pooler_config=PoolerConfig(task="token_embed"),
              max_model_len=4096, gpu_memory_utilization=0.85)
    os.makedirs(f"{ART}/offline", exist_ok=True)
    results = {}
    for p in IMAGE_PROBES:
        img = Image.open(f"/root/data/probes/{os.path.basename(p.path)}").convert("RGB")
        out = llm.encode([TextPrompt(prompt=build_image_prompt(), multi_modal_data={"image": img})])[0]
        hidden = out.outputs.data
        hidden = np.asarray(hidden.detach().cpu().numpy() if hasattr(hidden, "detach") else hidden,
                            dtype=np.float32)
        ids = np.asarray(out.prompt_token_ids, dtype=np.int64)
        save_artifact(f"{ART}/offline/{p.id}.npz", hidden, ids)
        results[p.id] = list(hidden.shape)
    artifacts.commit()
    print(results)
    return results
```

- [ ] **Step 3: Run both image harnesses**

Run:
```bash
uv run modal run src/jinav4_vllm/modal_app/reference.py::reference_image
uv run modal run src/jinav4_vllm/modal_app/offline.py::offline_image
```
Expected: both print `{image_cat: [n,128]/[n,2048], image_chart: ...}`.

- [ ] **Step 4: Spike R2 (image token alignment) via the parity report**

```bash
uv run modal volume get jinav4-artifacts /reference artifacts/ --force
uv run modal volume get jinav4-artifacts /offline   artifacts/ --force
uv run python -m jinav4_vllm.eval.report
```
Expected: image rows are **aligned ✅** (same image-token count) and `PASS`.
**If image rows show MISALIGNED:** this is the image-preprocessing drift risk (R2). Record the two token counts in `reports/image_alignment.md`. Mitigation: pin vLLM's image processor to match transformers (set identical `min_pixels`/`max_pixels`/`size` on both via the processor config), or pre-resize probe images to a fixed multiple of the patch size so both paths agree. Re-run until aligned. **Do not** truncate to force alignment.

- [ ] **Step 5: Commit**

```bash
git add src/jinav4_vllm/modal_app/reference.py src/jinav4_vllm/modal_app/offline.py reports/
git commit -m "feat: image reference + offline harnesses, R2 alignment check, Stage-1 image parity"
```

---

## Task 10: Variant B — server-side proxy (Modal) + Spike R1 + Stage 2 parity

The likely production shape: a FastAPI proxy co-located with vLLM that returns final `[n,128]`. Built before A/C because it both validates the served path and addresses R1 (multimodal over HTTP) with a controlled fallback.

**Files:**
- Create: `src/jinav4_vllm/modal_app/serve_b.py`
- Create: `src/jinav4_vllm/modal_app/client.py`

- [ ] **Step 1: Write the variant B server (FastAPI + in-process vLLM offline engine)**

> Design choice (R1 mitigation baked in): rather than depend on the undocumented multimodal `/pooling` HTTP path, variant B runs the vLLM engine **in-process** and exposes our own clean `/embed` endpoint. This is still "vLLM serving over HTTP," fully controls prompt construction (guaranteeing parity), and sidesteps R1. Variant A (Task 11) tests the native `vllm serve` HTTP path explicitly for comparison.

```python
# src/jinav4_vllm/modal_app/serve_b.py
"""Variant B: FastAPI + in-process vLLM engine; returns final [n,128] multivectors."""
from __future__ import annotations
import modal
from jinav4_vllm.modal_app.app import app, vllm_image, GPU, COMMON, ART

web_image = vllm_image.uv_pip_install("fastapi", "uvicorn", "pydantic")


@app.cls(image=web_image, gpu=GPU, timeout=3600, scaledown_window=600,
         min_containers=0, **COMMON)
@modal.concurrent(max_inputs=8)
class VariantB:
    @modal.enter()
    def load(self):
        import sys; sys.path.insert(0, "/root")
        import numpy as np
        from vllm import LLM
        from vllm.config import PoolerConfig
        self.np = np
        self.LLM = LLM
        self.llm = LLM(model="jinaai/jina-embeddings-v4-vllm-retrieval",
                       runner="pooling", pooler_config=PoolerConfig(task="token_embed"),
                       max_model_len=4096, gpu_memory_utilization=0.85)
        proj = np.load(f"{ART}/projector/retrieval.npz")
        self.W, self.b = proj["W"], proj["b"]
        from jinav4_vllm.multivector.core import to_multivector
        self.to_multivector = to_multivector

    def _encode(self, prompt_obj):
        out = self.llm.encode([prompt_obj])[0]
        hidden = out.outputs.data
        hidden = self.np.asarray(
            hidden.detach().cpu().numpy() if hasattr(hidden, "detach") else hidden,
            dtype=self.np.float32)
        mv = self.to_multivector(hidden, self.W, self.b)          # [n,128] server-side
        return mv, list(out.prompt_token_ids)

    @modal.asgi_app()
    def web(self):
        import base64, io
        from fastapi import FastAPI
        from pydantic import BaseModel
        from PIL import Image
        from vllm.inputs.data import TextPrompt
        from jinav4_vllm.common.probes import build_text_prompt, build_image_prompt

        api = FastAPI()

        class TextReq(BaseModel):
            text: str
            kind: str = "query"

        class ImageReq(BaseModel):
            image_b64: str

        @api.post("/embed/text")
        def embed_text(r: TextReq):
            mv, ids = self._encode(TextPrompt(prompt=build_text_prompt(r.text, r.kind)))
            return {"dim": 128, "tokens": mv.shape[0], "multivectors": mv.tolist(), "token_ids": ids}

        @api.post("/embed/image")
        def embed_image(r: ImageReq):
            img = Image.open(io.BytesIO(base64.b64decode(r.image_b64))).convert("RGB")
            mv, ids = self._encode(TextPrompt(prompt=build_image_prompt(), multi_modal_data={"image": img}))
            return {"dim": 128, "tokens": mv.shape[0], "multivectors": mv.tolist(), "token_ids": ids}

        return api
```

- [ ] **Step 2: Write the client that fills `variant_b` artifacts**

```python
# src/jinav4_vllm/modal_app/client.py
"""Hit a deployed variant endpoint and write artifacts/<variant>/<id>.npz locally."""
from __future__ import annotations
import base64, os, sys
import numpy as np
import requests
from jinav4_vllm.common.probes import TEXT_PROBES, IMAGE_PROBES
from jinav4_vllm.common.artifacts import save_artifact


def collect(base_url: str, variant: str, text_path: str, image_path: str):
    os.makedirs(f"artifacts/{variant}", exist_ok=True)
    for p in TEXT_PROBES:
        r = requests.post(f"{base_url}{text_path}", json={"text": p.text, "kind": p.kind}, timeout=120)
        r.raise_for_status(); d = r.json()
        save_artifact(f"artifacts/{variant}/{p.id}.npz",
                      np.asarray(d["multivectors"], np.float32), np.asarray(d["token_ids"], np.int64))
    for p in IMAGE_PROBES:
        b64 = base64.b64encode(open(p.path, "rb").read()).decode()
        r = requests.post(f"{base_url}{image_path}", json={"image_b64": b64}, timeout=120)
        r.raise_for_status(); d = r.json()
        save_artifact(f"artifacts/{variant}/{p.id}.npz",
                      np.asarray(d["multivectors"], np.float32), np.asarray(d["token_ids"], np.int64))
    print(f"wrote artifacts/{variant}/ from {base_url}")


if __name__ == "__main__":
    # usage: python -m jinav4_vllm.modal_app.client <variant> <base_url> <text_path> <image_path>
    collect(base_url=sys.argv[2], variant=sys.argv[1], text_path=sys.argv[3], image_path=sys.argv[4])
```

> `requests` is needed locally — add it to `[dependency-groups] dev` in `pyproject.toml` and `uv sync`.

- [ ] **Step 3: Deploy variant B and collect artifacts**

```bash
uv run modal deploy src/jinav4_vllm/modal_app/serve_b.py
# note the printed URL, e.g. https://<ws>--jinav4-vllm-variantb-web.modal.run
uv run python -m jinav4_vllm.modal_app.client variant_b <URL> /embed/text /embed/image
```
Expected: `wrote artifacts/variant_b/` with one npz per probe (text + image), each dim 128.

- [ ] **Step 4: Stage 2 parity for variant B**

```bash
uv run python -m jinav4_vllm.eval.report
```
Expected: `variant_b` rows present for text + image; `cos_mean ≥ 0.99`, aligned ✅, PASS. Since variant B applies the same projector + offline engine as Stage 1, it should match the offline artifacts to ~serialization epsilon.

- [ ] **Step 5: Commit**

```bash
git add src/jinav4_vllm/modal_app/serve_b.py src/jinav4_vllm/modal_app/client.py pyproject.toml reports/
git commit -m "feat: variant B server-side proxy + client + Stage-2 parity (R1 mitigated)"
```

---

## Task 11: Variant A — native `vllm serve` + client-side projection + Stage 2 parity

Tests the stock vLLM HTTP server (`/pooling`) with projection applied in the client — including the native multimodal HTTP path (R1) for real.

**Files:**
- Create: `src/jinav4_vllm/modal_app/serve_a.py`
- Modify: `src/jinav4_vllm/modal_app/client.py` (add a client-side-projection collector)

- [ ] **Step 1: Write the stock vLLM server on Modal**

```python
# src/jinav4_vllm/modal_app/serve_a.py
"""Variant A: stock `vllm serve --runner pooling` exposed via Modal web_server."""
from __future__ import annotations
import modal
from jinav4_vllm.modal_app.app import app, vllm_image, GPU, COMMON

VLLM_PORT = 8000


@app.function(image=vllm_image, gpu=GPU, timeout=3600, scaledown_window=600, **COMMON)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=VLLM_PORT, startup_timeout=600)
def serve_a():
    import subprocess
    cmd = [
        "vllm", "serve", "jinaai/jina-embeddings-v4-vllm-retrieval",
        "--runner", "pooling", "--pooler-config.task", "token_embed",
        "--served-model-name", "jina-v4", "--host", "0.0.0.0", "--port", str(VLLM_PORT),
        "--max-model-len", "4096", "--trust-remote-code",
    ]
    subprocess.Popen(" ".join(cmd), shell=True)
```

- [ ] **Step 2: Add the variant-A collector (raw `/pooling` → client projects)**

```python
# append to src/jinav4_vllm/modal_app/client.py

def collect_variant_a(base_url: str):
    """Variant A: server returns raw [n,2048] via /pooling; client applies the projector."""
    import base64
    from jinav4_vllm.multivector.core import to_multivector
    proj = np.load("artifacts/projector/retrieval.npz"); W, b = proj["W"], proj["b"]
    os.makedirs("artifacts/variant_a", exist_ok=True)

    def pool_text(prompt: str):
        r = requests.post(f"{base_url}/pooling", json={"model": "jina-v4", "input": [prompt]}, timeout=120)
        r.raise_for_status(); item = r.json()["data"][0]
        return np.asarray(item["data"], np.float32)            # [n,2048]

    def pool_image(b64: str):
        # native multimodal HTTP path (R1): chat-style messages with image_url to /pooling
        msg = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": "Describe the image."}]}]
        r = requests.post(f"{base_url}/pooling", json={"model": "jina-v4", "messages": msg}, timeout=120)
        r.raise_for_status(); item = r.json()["data"][0]
        return np.asarray(item["data"], np.float32)

    from jinav4_vllm.common.probes import TEXT_PROBES, IMAGE_PROBES, build_text_prompt
    for p in TEXT_PROBES:
        hidden = pool_text(build_text_prompt(p.text, p.kind))
        # token_ids unavailable over /pooling → store an index range; alignment vs reference
        # is asserted via token COUNT for variant A (see report note).
        ids = np.arange(hidden.shape[0], dtype=np.int64)
        save_artifact(f"artifacts/variant_a/{p.id}.npz", to_multivector(hidden, W, b), ids)
    for p in IMAGE_PROBES:
        b64 = base64.b64encode(open(p.path, "rb").read()).decode()
        try:
            hidden = pool_image(b64)
            ids = np.arange(hidden.shape[0], dtype=np.int64)
            save_artifact(f"artifacts/variant_a/{p.id}.npz", to_multivector(hidden, W, b), ids)
        except requests.HTTPError as e:
            print(f"[R1] native multimodal /pooling failed for {p.id}: {e} "
                  "-> variant A does not support image over native HTTP; documented in report.")
    print("wrote artifacts/variant_a/")
```

**Variant A loses real token_ids over `/pooling`** (the endpoint returns only the matrix), so for that source we align on token **count**, not ids. Make this concrete:

Modify `run_comparison` in `src/jinav4_vllm/eval/report.py` to accept `count_only_sources` and branch the alignment check:

```python
# src/jinav4_vllm/eval/report.py  — replace the run_comparison signature + alignment block
def run_comparison(ref_dir: str, source_dirs: dict[str, str],
                   W: np.ndarray | None, b: np.ndarray | None,
                   probe_ids: list[str],
                   count_only_sources: set[str] | None = None) -> list[dict]:
    count_only_sources = count_only_sources or set()
    rows: list[dict] = []
    for pid in probe_ids:
        ref_path = os.path.join(ref_dir, f"{pid}.npz")
        if not os.path.exists(ref_path):
            continue
        ref_mv, ref_ids = materialize(ref_path, W, b)
        for source, d in source_dirs.items():
            ap = os.path.join(d, f"{pid}.npz")
            if not os.path.exists(ap):
                continue
            mv, ids = materialize(ap, W, b)
            row = {"probe_id": pid, "source": source}
            try:
                if source in count_only_sources:
                    if ids.shape[0] != ref_ids.shape[0]:
                        raise AlignmentError(
                            f"length {ids.shape[0]} vs {ref_ids.shape[0]}")
                else:
                    assert_aligned(ids, ref_ids)
                row["aligned"] = True
                row.update(compare(mv, ref_mv))
            except AlignmentError as e:
                row.update({"aligned": False, "error": str(e),
                            "ref_tokens": int(ref_ids.shape[0]),
                            "var_tokens": int(ids.shape[0])})
            rows.append(row)
    return rows
```

And in `main()`, pass `count_only_sources={"variant_a"}`:

```python
    rows = run_comparison("artifacts/reference", sources, W, b, probe_ids,
                          count_only_sources={"variant_a", "variant_c"})
```

Extend `tests/test_eval_report.py` with a count-only case:

```python
def test_run_comparison_count_only_aligns_on_length(tmp_path):
    import numpy as np
    from jinav4_vllm.eval.report import run_comparison
    from jinav4_vllm.common.artifacts import save_artifact
    from jinav4_vllm.multivector.core import to_multivector
    rng = np.random.default_rng(7)
    hidden = rng.standard_normal((5, 2048)).astype(np.float32)
    W = rng.standard_normal((128, 2048)).astype(np.float32)
    b = rng.standard_normal((128,)).astype(np.float32)
    ref_mv = to_multivector(hidden, W, b)
    ref = tmp_path / "reference"; va = tmp_path / "variant_a"; ref.mkdir(); va.mkdir()
    save_artifact(str(ref / "x.npz"), ref_mv, np.array([10, 11, 12, 13, 14], dtype=np.int64))
    # variant_a stores final 128-dim with placeholder index ids but same COUNT
    save_artifact(str(va / "x.npz"), ref_mv, np.arange(5, dtype=np.int64))
    rows = run_comparison(str(ref), {"variant_a": str(va)}, W, b, ["x"],
                          count_only_sources={"variant_a"})
    assert rows[0]["aligned"] is True
    assert rows[0]["cosine_min"] > 0.9999
```

Run: `uv run pytest tests/test_eval_report.py -q` → PASS before deploying.

- [ ] **Step 3: Deploy and collect**

```bash
uv run modal deploy src/jinav4_vllm/modal_app/serve_a.py
uv run python -c "from jinav4_vllm.modal_app.client import collect_variant_a; collect_variant_a('<URL>')"
```
Expected: `artifacts/variant_a/` written for text (and image if R1 path works). If the image `/pooling` call fails, that's a **documented R1 finding** — variant A cannot serve images over native HTTP without a custom chat template; note it and move on.

- [ ] **Step 4: Stage 2 parity for variant A**

```bash
uv run python -m jinav4_vllm.eval.report
```
Expected: `variant_a` text rows PASS with `cos_mean ≥ 0.99`. This row specifically measures **serialization precision** of the 2048-dim intermediate over JSON (the spec's variant-A hypothesis). Compare its `max_abs_diff` vs variant B's to quantify the wire-precision loss.

- [ ] **Step 5: Commit**

```bash
git add src/jinav4_vllm/modal_app/serve_a.py src/jinav4_vllm/modal_app/client.py src/jinav4_vllm/eval/report.py tests/test_eval_report.py reports/
git commit -m "feat: variant A native vllm serve + client projection + Stage-2 parity (R1 tested)"
```

---

## Task 12: Variant C — in-vLLM pooler plugin (time-boxed) + Stage 2 parity

Make native `/pooling` emit final `[n,128]` by registering a custom pooling model that loads the projector in-engine. **Time-box to one focused attempt on the pinned vLLM version (R5);** if infeasible, document and proceed.

**Files:**
- Create: `src/jinav4_vllm/modal_app/serve_c.py`
- Create: `src/jinav4_vllm/plugin/jina_v4_multivector.py` (the custom model/pooler)

- [ ] **Step 1: Write the custom model plugin**

```python
# src/jinav4_vllm/plugin/jina_v4_multivector.py
"""Out-of-tree vLLM model that wraps the Qwen2.5-VL pooling model and applies the
retrieval multi_vector_projector in-engine, so /pooling returns final [n,128].

Registered via vLLM's ModelRegistry in serve_c.py. APIs here target the pinned
vLLM version; if class/hook names differ, adjust to the version's pooling model
interface (VllmModelForPooling) confirmed in Task 5.
"""
from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn


class JinaV4MultiVector(nn.Module):
    """Thin wrapper: delegate the backbone to the registered Qwen2.5-VL pooling model,
    then project + normalize per token inside the pooler."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration as Base
        self.backbone = Base(*args, **kwargs)
        proj = np.load(os.environ["PROJECTOR_NPZ"])
        self.register_buffer("W", torch.tensor(proj["W"], dtype=torch.float16))   # [128,2048]
        self.register_buffer("b", torch.tensor(proj["b"], dtype=torch.float16))   # [128]

    def __getattr__(self, name):
        # delegate weight loading / forward plumbing to the backbone
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.__dict__["_modules"]["backbone"], name)

    def pooler(self, hidden_states: torch.Tensor, pooling_metadata):
        # hidden_states: [total_tokens, 2048] for ALL/token_embed pooling
        projected = hidden_states.to(self.W.dtype) @ self.W.T + self.b
        projected = torch.nn.functional.normalize(projected, dim=-1)
        # Re-use the backbone's pooler envelope to keep output formatting identical,
        # substituting our projected states. If the version exposes a PoolerOutput
        # builder, wrap `projected` accordingly here.
        return projected
```

- [ ] **Step 2: Write the variant C server**

```python
# src/jinav4_vllm/modal_app/serve_c.py
"""Variant C: native vllm serve with the custom JinaV4MultiVector model registered."""
from __future__ import annotations
import modal
from jinav4_vllm.modal_app.app import app, vllm_image, GPU, COMMON, ART

VLLM_PORT = 8001
plugin_image = vllm_image  # plugin code is on the src mount


@app.function(image=plugin_image, gpu=GPU, timeout=3600, scaledown_window=600,
              env={"PROJECTOR_NPZ": f"{ART}/projector/retrieval.npz"}, **COMMON)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=VLLM_PORT, startup_timeout=600)
def serve_c():
    import subprocess, textwrap, os
    # Register the custom architecture, then launch the API server in-process.
    reg = textwrap.dedent("""
        import sys; sys.path.insert(0, "/root")
        from vllm import ModelRegistry
        from jinav4_vllm.plugin.jina_v4_multivector import JinaV4MultiVector
        ModelRegistry.register_model("JinaV4MultiVector", JinaV4MultiVector)
        from vllm.entrypoints.openai.api_server import main
        main()
    """)
    os.makedirs("/root/_c", exist_ok=True)
    with open("/root/_c/launch.py", "w") as f:
        f.write(reg)
    cmd = ["python", "/root/_c/launch.py", "serve",
           "jinaai/jina-embeddings-v4-vllm-retrieval",
           "--runner", "pooling", "--pooler-config.task", "token_embed",
           "--served-model-name", "jina-v4", "--host", "0.0.0.0", "--port", str(VLLM_PORT),
           "--max-model-len", "4096", "--trust-remote-code",
           "--hf-overrides", '{"architectures":["JinaV4MultiVector"]}']
    subprocess.Popen(" ".join(cmd), shell=True)
```

- [ ] **Step 3: Deploy, smoke-test the dimension, collect**

```bash
uv run modal deploy src/jinav4_vllm/modal_app/serve_c.py
# smoke: one /pooling call must return dim 128, not 2048
uv run python -c "import requests,numpy as np; \
d=requests.post('<URL>/pooling', json={'model':'jina-v4','input':['Query: hi']}).json(); \
print('dim', np.array(d['data'][0]['data']).shape)"
```
Expected: `dim (N, 128)`.
**R5 time-box:** if registration/`pooler` hooks don't fit the pinned vLLM interface after one focused attempt, stop. Write `reports/variant_c.md` documenting the blocker (exact API mismatch) and conclude "C not viable on vllm==<ver>; recommend B." Skip Steps 4–5.

- [ ] **Step 4: Collect variant-C artifacts (server already returns 128-dim)**

Append to `src/jinav4_vllm/modal_app/client.py` (the server already projects in-engine, so the client saves the matrix directly — no client-side projection):

```python
# append to src/jinav4_vllm/modal_app/client.py

def collect_variant_c(base_url: str):
    """Variant C: /pooling returns final [n,128] (projection done in-engine)."""
    import base64
    from jinav4_vllm.common.probes import TEXT_PROBES, IMAGE_PROBES, build_text_prompt
    os.makedirs("artifacts/variant_c", exist_ok=True)

    def pool_text(prompt: str):
        r = requests.post(f"{base_url}/pooling", json={"model": "jina-v4", "input": [prompt]}, timeout=120)
        r.raise_for_status(); return np.asarray(r.json()["data"][0]["data"], np.float32)

    def pool_image(b64: str):
        msg = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": "Describe the image."}]}]
        r = requests.post(f"{base_url}/pooling", json={"model": "jina-v4", "messages": msg}, timeout=120)
        r.raise_for_status(); return np.asarray(r.json()["data"][0]["data"], np.float32)

    for p in TEXT_PROBES:
        mv = pool_text(build_text_prompt(p.text, p.kind))
        save_artifact(f"artifacts/variant_c/{p.id}.npz", mv, np.arange(mv.shape[0], dtype=np.int64))
    for p in IMAGE_PROBES:
        b64 = base64.b64encode(open(p.path, "rb").read()).decode()
        try:
            mv = pool_image(b64)
            save_artifact(f"artifacts/variant_c/{p.id}.npz", mv, np.arange(mv.shape[0], dtype=np.int64))
        except requests.HTTPError as e:
            print(f"[variant_c] image over /pooling failed for {p.id}: {e}")
    print("wrote artifacts/variant_c/")
```

Run: `uv run python -c "from jinav4_vllm.modal_app.client import collect_variant_c; collect_variant_c('<URL>')"`
Expected: `artifacts/variant_c/` written; each text npz is `[n,128]`. Variant C aligns on token **count** (no token_ids over `/pooling`); `report.main()` already includes `"variant_c"` in `count_only_sources` from Task 11.

- [ ] **Step 5: Stage 2 parity for variant C, then commit**

```bash
uv run python -m jinav4_vllm.eval.report
git add src/jinav4_vllm/plugin/ src/jinav4_vllm/modal_app/serve_c.py reports/
git commit -m "feat: variant C in-vLLM plugin + Stage-2 parity (or documented infeasibility)"
```
Expected: `variant_c` rows present (or `reports/variant_c.md` explains why not). Variant C should be the most faithful (no intermediate serialization; in-engine fp16 projection).

---

## Task 13: Comparative report + production recommendation

**Files:**
- Create: `src/jinav4_vllm/eval/recommend.py`
- Create: `reports/RECOMMENDATION.md` (generated)

- [ ] **Step 1: Write the recommendation generator**

```python
# src/jinav4_vllm/eval/recommend.py
"""Summarize parity.json across variants and emit a production recommendation."""
from __future__ import annotations
import json, os
from collections import defaultdict


def summarize(parity_json: str = "reports/parity.json") -> str:
    rows = json.load(open(parity_json))
    by_src = defaultdict(list)
    for r in rows:
        if r.get("aligned"):
            by_src[r["source"]].append(r)
    lines = ["# Production Recommendation", "",
             "## Parity summary (lower diff / higher cosine = closer to reference)", "",
             "| variant | probes | worst cos_min | mean cos_mean | worst max_abs_diff |",
             "|---|---|---|---|---|"]
    stats = {}
    for src, rs in sorted(by_src.items()):
        cos_min = min(r["cosine_min"] for r in rs)
        cos_mean = sum(r["cosine_mean"] for r in rs) / len(rs)
        max_abs = max(r["max_abs_diff"] for r in rs)
        stats[src] = (cos_min, cos_mean, max_abs)
        lines += [f"| {src} | {len(rs)} | {cos_min:.5f} | {cos_mean:.5f} | {max_abs:.2e} |"]
    lines += ["", "## Operational trade-offs", "",
              "| variant | wire payload | client complexity | server complexity | multimodal HTTP | version-fragility |",
              "|---|---|---|---|---|---|",
              "| variant_a | raw 2048/token (large) | high (carries projector) | none (stock) | native /pooling (R1 result) | low |",
              "| variant_b | final 128/token (small) | none | moderate (FastAPI+engine) | in-process (controlled) | low |",
              "| variant_c | final 128/token (small) | none | high (vLLM plugin) | native | high (R5) |",
              "", "## Recommendation", ""]
    # Heuristic: prefer the variant that (a) passes parity and (b) minimizes ops risk.
    note = ("Default to **variant B** unless variant C demonstrably beats it on parity AND is "
            "viable on the pinned vLLM version. Variant A is the fallback if a fully-stock vLLM "
            "image is a hard requirement and clients can carry the projector.")
    lines.append(note)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    md = summarize()
    os.makedirs("reports", exist_ok=True)
    open("reports/RECOMMENDATION.md", "w").write(md)
    print(md)
```

- [ ] **Step 2: Generate the recommendation**

Run: `uv run python -m jinav4_vllm.eval.recommend`
Expected: prints the comparative table + recommendation; writes `reports/RECOMMENDATION.md`.

- [ ] **Step 3: Final full-suite check**

Run: `uv run pytest -q`
Expected: all local tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/jinav4_vllm/eval/recommend.py reports/
git commit -m "feat: comparative report + production recommendation"
```

---

## Self-Review notes (for the implementer)

- **Reference fidelity:** Tasks 6/9 trust the high-level `encode_text/encode_image(return_multivector=True)` as ground truth and derive token_ids from the identical prompt. The `assert mv.shape[0] == ids.shape[0]` is the guard that the token sequence we align on truly matches the reference's internal sequence. If it ever fails, fix tokenization before trusting any parity number.
- **Threshold honesty:** the report applies `cos_mean ≥ 0.99 / max_abs ≤ 1e-3`. Record the *achieved* fp16 floor in `reports/`; if the real floor is e.g. 0.997 across all variants, state that rather than silently relaxing — that floor IS the feasibility finding.
- **R1/R2/R5 are findings, not just risks:** each has a designated report file. A "negative" result (e.g. native multimodal `/pooling` unsupported, or variant C infeasible) is a valid, valuable study outcome — document it; don't paper over it.
- **Version pinning:** Task 5 Step 3 pins vLLM. All later vLLM code assumes that pinned API (`runner="pooling"`, `PoolerConfig(task="token_embed")`). If the spike chose a fallback API, propagate that exact form to Tasks 7, 9, 10, 11, 12.
```
