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
