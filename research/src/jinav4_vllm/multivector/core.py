# src/jinav4_vllm/multivector/core.py
"""Pure-NumPy multi-vector post-processing — the single source of projection math.

Per the spec, multi-vector mode keeps all non-pad tokens (text: incl.
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
