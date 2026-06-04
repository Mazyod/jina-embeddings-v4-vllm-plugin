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
