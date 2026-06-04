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
