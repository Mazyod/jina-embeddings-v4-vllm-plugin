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

def test_run_comparison_count_only_aligns_on_length(tmp_path):
    from jinav4_vllm.eval.report import run_comparison
    from jinav4_vllm.multivector.core import to_multivector
    rng = np.random.default_rng(7)
    hidden = rng.standard_normal((5, 2048)).astype(np.float32)
    W = rng.standard_normal((128, 2048)).astype(np.float32)
    b = rng.standard_normal((128,)).astype(np.float32)
    ref_mv = to_multivector(hidden, W, b)
    ref = tmp_path / "reference"; sv = tmp_path / "served"; ref.mkdir(); sv.mkdir()
    save_artifact(str(ref / "x.npz"), ref_mv, np.array([10, 11, 12, 13, 14], dtype=np.int64))
    # served stores final 128-dim with placeholder index ids but same COUNT (/pooling drops ids)
    save_artifact(str(sv / "x.npz"), ref_mv, np.arange(5, dtype=np.int64))
    rows = run_comparison(str(ref), {"served": str(sv)}, W, b, ["x"],
                          count_only_sources={"served"})
    assert rows[0]["aligned"] is True
    assert rows[0]["cosine_min"] > 0.9999
