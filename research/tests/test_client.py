# tests/test_client.py
import numpy as np
from jinav4_vllm.client import JinaV4Client


class _FakeResp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


class _FakeSession:
    """Records POSTs and returns a fixed /pooling-shaped payload."""

    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def post(self, url, json, timeout):
        self.calls.append((url, json))
        return _FakeResp({"data": [{"data": self.rows, "index": 0}]})


def test_embed_text_query_prefix_and_shape():
    rows = [[0.0] * 128, [1.0] + [0.0] * 127]
    sess = _FakeSession(rows)
    out = JinaV4Client("http://x", session=sess).embed_text("hello", "query")
    assert out.shape == (2, 128)
    url, body = sess.calls[0]
    assert url == "http://x/pooling"
    assert body["model"] == "jina-v4"
    assert body["input"] == ["Query: hello"]


def test_embed_text_passage_prefix_and_url_normalization():
    sess = _FakeSession([[0.0] * 128])
    JinaV4Client("http://x/", session=sess).embed_text("doc", "passage")
    url, body = sess.calls[0]
    assert url == "http://x/pooling"          # trailing slash stripped
    assert body["input"] == ["Passage: doc"]


def test_embed_image_bytes_builds_data_url():
    sess = _FakeSession([[0.0] * 128])
    JinaV4Client("http://x", session=sess).embed_image(b"\x89PNG-fake-bytes")
    content = sess.calls[0][1]["messages"][0]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1]["text"] == "Describe the image."


def test_embed_image_passthrough_url_not_reencoded():
    sess = _FakeSession([[0.0] * 128])
    JinaV4Client("http://x", session=sess).embed_image("https://host/page.png")
    url = sess.calls[0][1]["messages"][0]["content"][0]["image_url"]["url"]
    assert url == "https://host/page.png"


def test_maxsim_matches_core():
    from jinav4_vllm.multivector.core import maxsim
    q = np.eye(3, 128, dtype=np.float32)
    d = np.eye(4, 128, dtype=np.float32)
    assert JinaV4Client.maxsim(q, d) == maxsim(q, d)
