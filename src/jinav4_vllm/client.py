"""JinaV4Client — minimal HTTP client for the vLLM multi-vector `/pooling` endpoint.

Talks to a stock vLLM OpenAI server running the JinaV4MultiVector plugin (see deploy/DEPLOY.md).
`/pooling` returns final L2-normalized `[n,128]` per-token multivectors for text and images, so
this client does no projection math — it only formats requests and parses the matrix back.

    from jinav4_vllm.client import JinaV4Client
    c = JinaV4Client("http://localhost:8000")
    q = c.embed_text("climate impact on coastal cities", "query")   # [n,128]
    d = c.embed_image("page.png")                                   # [m,128]
    score = c.maxsim(q, d)                                          # ColBERT late interaction
"""
from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import requests

from jinav4_vllm.multivector.core import maxsim as _maxsim

DEFAULT_MODEL = "jina-v4"


class JinaV4Client:
    """Thin client over the served `/pooling` endpoint. Vectors come back L2-normalized."""

    def __init__(self, base_url: str, model: str = DEFAULT_MODEL, timeout: float = 180.0,
                 session: requests.Session | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._session = session or requests.Session()

    def _post_pooling(self, payload: dict) -> np.ndarray:
        r = self._session.post(f"{self.base_url}/pooling",
                               json={"model": self.model, **payload}, timeout=self.timeout)
        r.raise_for_status()
        return np.asarray(r.json()["data"][0]["data"], dtype=np.float32)

    def embed_text(self, text: str, kind: str = "query") -> np.ndarray:
        """Embed text. `kind` ∈ {"query","passage"} applies Jina's retrieval prefix. Returns [n,128]."""
        prefix = {"query": "Query", "passage": "Passage"}[kind]
        return self._post_pooling({"input": [f"{prefix}: {text}"]})

    def embed_image(self, image: str | bytes | Path,
                    prompt: str = "Describe the image.") -> np.ndarray:
        """Embed an image given a path, raw bytes, or a data:/http(s): URL. Returns [m,128]."""
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": self._image_url(image)}},
            {"type": "text", "text": prompt}]}]
        return self._post_pooling({"messages": messages})

    @staticmethod
    def _image_url(image: str | bytes | Path) -> str:
        if isinstance(image, (str, Path)) and str(image).startswith(("http://", "https://", "data:")):
            return str(image)
        data = image if isinstance(image, bytes) else Path(image).read_bytes()
        return "data:image/png;base64," + base64.b64encode(data).decode()

    @staticmethod
    def maxsim(query_mv: np.ndarray, doc_mv: np.ndarray) -> float:
        """ColBERT late-interaction score. Vectors are already L2-normalized, so this is plain dots."""
        return _maxsim(query_mv, doc_mv)
