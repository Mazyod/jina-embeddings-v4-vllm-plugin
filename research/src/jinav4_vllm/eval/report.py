# src/jinav4_vllm/eval/report.py
"""Materialize artifacts to [n,128] and compare each variant to the reference."""
from __future__ import annotations
import os, json
import numpy as np
from jinav4_vllm.common.artifacts import load_artifact
from jinav4_vllm.multivector.core import to_multivector
from jinav4_vllm.eval.metrics import assert_aligned, compare, AlignmentError


def materialize(path: str, W: np.ndarray | None, b: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Load an artifact and return final [n,128] multivectors. Projects if stored at 2048."""
    mv, ids = load_artifact(path)
    if mv.shape[1] == 2048:
        assert W is not None and b is not None, "need projector to materialize 2048-dim artifact"
        mv = to_multivector(mv, W, b)
    return mv, ids


def run_comparison(ref_dir: str, source_dirs: dict[str, str],
                   W: np.ndarray | None, b: np.ndarray | None,
                   probe_ids: list[str],
                   count_only_sources: set[str] | None = None) -> list[dict]:
    count_only_sources = count_only_sources or set()
    rows: list[dict] = []
    for pid in probe_ids:
        ref_path = os.path.join(ref_dir, f"{pid}.npz")
        if not os.path.exists(ref_path):
            continue
        ref_mv, ref_ids = materialize(ref_path, W, b)
        for source, d in source_dirs.items():
            ap = os.path.join(d, f"{pid}.npz")
            if not os.path.exists(ap):
                continue
            mv, ids = materialize(ap, W, b)
            row = {"probe_id": pid, "source": source}
            try:
                if source in count_only_sources:
                    if ids.shape[0] != ref_ids.shape[0]:
                        raise AlignmentError(
                            f"length {ids.shape[0]} vs {ref_ids.shape[0]}")
                else:
                    assert_aligned(ids, ref_ids)
                row["aligned"] = True
                row.update(compare(mv, ref_mv))
            except AlignmentError as e:
                row.update({"aligned": False, "error": str(e),
                            "ref_tokens": int(ref_ids.shape[0]),
                            "var_tokens": int(ids.shape[0])})
            rows.append(row)
    return rows


def render_markdown(rows: list[dict], thresholds: dict) -> str:
    lines = ["# Parity Report", "", "| probe | source | aligned | n | max_abs | mean_abs | cos_min | cos_mean | relF | pass |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if not r.get("aligned", False):
            lines.append(f"| {r['probe_id']} | {r['source']} | ❌ | - | - | - | - | - | - | "
                         f"MISALIGNED ({r.get('ref_tokens')}≠{r.get('var_tokens')}) |")
            continue
        # Gate on aggregate per-token cosine (direction agreement). cos_min is reported
        # informationally: image-patch (token 151655) embeddings carry a larger bf16
        # vision-encoder floor, so a few patches dip while the mean stays >=0.99.
        ok = r["cosine_mean"] >= thresholds["cosine_mean"]
        lines.append(f"| {r['probe_id']} | {r['source']} | ✅ | {r['n_tokens']} | "
                     f"{r['max_abs_diff']:.2e} | {r['mean_abs_diff']:.2e} | {r['cosine_min']:.5f} | "
                     f"{r['cosine_mean']:.5f} | {r['rel_frobenius']:.2e} | {'PASS' if ok else 'FAIL'} |")
    return "\n".join(lines) + "\n"


def main():
    """CLI: compares artifacts/ pulled from the Modal volume. Stage 1 = offline vs reference."""
    proj = np.load("artifacts/projector/retrieval.npz")
    W, b = proj["W"], proj["b"]
    from jinav4_vllm.common.probes import TEXT_PROBES, IMAGE_PROBES
    probe_ids = [p.id for p in TEXT_PROBES] + [p.id for p in IMAGE_PROBES]
    # offline = vLLM in-process baseline; served = the deployed vLLM OpenAI server (/pooling).
    sources = {name: f"artifacts/{name}" for name in ("offline", "served")
               if os.path.isdir(f"artifacts/{name}")}
    # served is count-only: /pooling returns the [n,128] matrix without token_ids, so we align on
    # row count (offline carries true prompt_token_ids and aligns by id).
    rows = run_comparison("artifacts/reference", sources, W, b, probe_ids,
                          count_only_sources={"served"})
    # Gate on aggregate per-token cosine (direction is what late-interaction MaxSim uses).
    # Achieved floors (bf16; canonical Jina encode_* is bf16, vLLM backbone is bf16 with
    # different kernels): text cos_mean ~0.999 / cos_min ~0.994; image cos_mean ~0.992-0.997
    # with per-patch cos_min outliers (vision-encoder bf16). See docs/VALIDATION.md.
    thresholds = {"cosine_mean": 0.99}
    md = render_markdown(rows, thresholds)
    os.makedirs("reports", exist_ok=True)
    with open("reports/parity.md", "w") as f: f.write(md)
    with open("reports/parity.json", "w") as f: json.dump(rows, f, indent=2)
    print(md)


if __name__ == "__main__":
    main()
