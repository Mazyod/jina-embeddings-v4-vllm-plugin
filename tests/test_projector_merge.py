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
