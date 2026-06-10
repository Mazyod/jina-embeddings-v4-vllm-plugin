"""Shared image-fidelity controls for Qwen2.5-VL / Jina v4 (dynamic resolution).

Image tokens ≈ (resized_H · resized_W) / (28·28·merge). Raise `max_pixels` for higher fidelity
(more patch tokens) at more compute. `pixels` = resized H·W.

Defaults are UNSET so the checkpoint's own `preprocessor_config.json` values apply (keeps the
verified parity demo green). Override per process via env vars; for a baked checkpoint, bake the
values into `preprocessor_config.json` instead (see deploy/bake_checkpoint.py --min/--max-pixels).

IMPORTANT: reference and served paths must use the SAME values or image-token counts diverge and
the per-token parity alignment check fails.
"""
from __future__ import annotations
import os

ENV_MIN = "JINA_IMAGE_MIN_PIXELS"
ENV_MAX = "JINA_IMAGE_MAX_PIXELS"

# Handy reference points (Qwen2.5-VL convention): pixels are multiples of 28·28 (=784).
QWEN_PATCH_AREA = 28 * 28
PRESET_MIN = 256 * QWEN_PATCH_AREA      # 200_704  (~448x448)
PRESET_MAX_STANDARD = 1280 * QWEN_PATCH_AREA   # 1_003_520 (Qwen2.5-VL default ceiling)
PRESET_MAX_HIFI = 4096 * QWEN_PATCH_AREA       # 3_211_264 (high-fidelity documents)


def mm_processor_kwargs(min_pixels: int = 0, max_pixels: int = 0) -> dict:
    """Return {min_pixels, max_pixels}, or {} to use the checkpoint defaults.

    Resolution order per bound: explicit arg (> 0) wins, else the env var (JINA_IMAGE_*), else unset.
    Explicit args matter for Modal: local env vars are NOT forwarded to remote containers, so the
    reference/offline jobs take these as `modal run … --min-pixels/--max-pixels` parameters.
    """
    def pick(arg: int, env: str) -> int:
        if arg:
            return int(arg)
        v = os.environ.get(env)
        return int(v) if v else 0

    kw: dict[str, int] = {}
    if (mn := pick(min_pixels, ENV_MIN)):
        kw["min_pixels"] = mn
    if (mx := pick(max_pixels, ENV_MAX)):
        kw["max_pixels"] = mx
    return kw
