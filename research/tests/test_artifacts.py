import numpy as np
from jinav4_vllm.common.artifacts import save_artifact, load_artifact

def test_roundtrip(tmp_path):
    mv = np.random.default_rng(0).standard_normal((5, 128)).astype(np.float32)
    ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    p = tmp_path / "probe.npz"
    save_artifact(str(p), mv, ids)
    mv2, ids2 = load_artifact(str(p))
    np.testing.assert_array_equal(mv, mv2)
    np.testing.assert_array_equal(ids, ids2)
    assert mv2.dtype == np.float32 and ids2.dtype == np.int64
