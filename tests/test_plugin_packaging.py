"""Guard the plugin's packaging contract without importing vLLM (CI-friendly, no GPU).

These assert the wiring a stock `vllm serve` relies on: the entry point, the registered model
target, the chat template shipping with the package, and the deliberate no-runtime-deps invariant
that keeps `pip install --no-deps` working inside the official vLLM image.
"""
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "jina_v4_vllm_plugin"


def _pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_entry_point_registers_plugin():
    eps = _pyproject()["project"]["entry-points"]["vllm.general_plugins"]
    assert eps == {"jina_v4": "jina_v4_vllm_plugin:register"}


def test_register_targets_model_class():
    src = (PKG / "__init__.py").read_text()
    assert "JinaV4MultiVector" in src
    assert "jina_v4_vllm_plugin.model:JinaV4MultiVectorModel" in src


def test_chat_template_ships_with_package():
    # hatchling bundles every file under the package directory; assert it is present at the
    # packaged path (the wheel-contents check lives in the build step / CI).
    assert (PKG / "jina_image_chat_template.jinja").exists()


def test_chat_template_has_vision_tokens():
    tmpl = (PKG / "jina_image_chat_template.jinja").read_text()
    for tok in ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>"):
        assert tok in tmpl


def test_no_runtime_dependencies():
    # vLLM/torch must NOT be hard deps so `pip install --no-deps` does not re-resolve them.
    assert _pyproject()["project"].get("dependencies", []) == []
