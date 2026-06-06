# tests/test_plugin_packaging.py
"""Guard the plugin's packaging contract without importing vLLM (CI-friendly, no GPU).

These assert the wiring a stock `vllm serve` relies on: the entry point, the registered model
target, and that the chat template ships in the wheel with the right vision tokens.
"""
import tomllib
from pathlib import Path

PLUGIN = Path(__file__).resolve().parents[1] / "src/jinav4_vllm/vllm_plugin"
PKG = PLUGIN / "jina_v4_vllm_plugin"


def _pyproject():
    return tomllib.loads((PLUGIN / "pyproject.toml").read_text())


def test_entry_point_registers_plugin():
    eps = _pyproject()["project"]["entry-points"]["vllm.general_plugins"]
    assert eps == {"jina_v4": "jina_v4_vllm_plugin:register"}


def test_register_targets_model_class():
    src = (PKG / "__init__.py").read_text()
    assert "JinaV4MultiVector" in src
    assert "jina_v4_vllm_plugin.model:JinaV4MultiVectorModel" in src


def test_chat_template_ships_in_wheel():
    pd = _pyproject()["tool"]["setuptools"]["package-data"]["jina_v4_vllm_plugin"]
    assert any(g == "*.jinja" or g.endswith(".jinja") for g in pd)
    assert (PKG / "jina_image_chat_template.jinja").exists()


def test_chat_template_has_vision_tokens():
    tmpl = (PKG / "jina_image_chat_template.jinja").read_text()
    for tok in ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>"):
        assert tok in tmpl
