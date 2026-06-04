"""Persist/read multivector artifacts as .npz: mv [n,D] float32, token_ids [n] int64."""
from __future__ import annotations
import numpy as np


def save_artifact(path: str, mv: np.ndarray, token_ids: np.ndarray) -> None:
    np.savez(path, mv=mv.astype(np.float32), token_ids=np.asarray(token_ids, dtype=np.int64))


def load_artifact(path: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    return d["mv"].astype(np.float32), d["token_ids"].astype(np.int64)
