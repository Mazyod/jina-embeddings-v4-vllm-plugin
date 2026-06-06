import importlib
import jinav4_vllm.common.imaging as im


def test_unset_returns_empty(monkeypatch):
    monkeypatch.delenv(im.ENV_MIN, raising=False)
    monkeypatch.delenv(im.ENV_MAX, raising=False)
    importlib.reload(im)
    assert im.mm_processor_kwargs() == {}


def test_env_overrides_are_read(monkeypatch):
    monkeypatch.setenv(im.ENV_MIN, "200704")
    monkeypatch.setenv(im.ENV_MAX, "3211264")
    assert im.mm_processor_kwargs() == {"min_pixels": 200704, "max_pixels": 3211264}


def test_presets_are_patch_multiples():
    for v in (im.PRESET_MIN, im.PRESET_MAX_STANDARD, im.PRESET_MAX_HIFI):
        assert v % im.QWEN_PATCH_AREA == 0


def test_explicit_args_override_env(monkeypatch):
    monkeypatch.setenv(im.ENV_MIN, "200704")
    monkeypatch.setenv(im.ENV_MAX, "3211264")
    # explicit args win over env for each bound
    assert im.mm_processor_kwargs(min_pixels=1003520) == {
        "min_pixels": 1003520, "max_pixels": 3211264}
    assert im.mm_processor_kwargs(min_pixels=1003520, max_pixels=4014080) == {
        "min_pixels": 1003520, "max_pixels": 4014080}


def test_explicit_args_without_env(monkeypatch):
    monkeypatch.delenv(im.ENV_MIN, raising=False)
    monkeypatch.delenv(im.ENV_MAX, raising=False)
    assert im.mm_processor_kwargs(max_pixels=3211264) == {"max_pixels": 3211264}
    assert im.mm_processor_kwargs() == {}
