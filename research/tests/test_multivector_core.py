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
