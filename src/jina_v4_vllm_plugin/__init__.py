"""vLLM general plugin: register the JinaV4MultiVector architecture in every vLLM process.

Installed as `jina-v4-vllm-plugin`; declared as a `vllm.general_plugins` entry point so a stock
`vllm serve` loads `register()` in every process (incl. the v1 EngineCore worker) and the
`/pooling` endpoint returns final 128-dim multivectors. The image chat template ships inside this
package — get its path with `chat_template_path()`.
"""
from importlib import resources
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("jina-v4-vllm-plugin")
except PackageNotFoundError:  # running from a source checkout that isn't installed
    __version__ = "0+unknown"

CHAT_TEMPLATE_FILE = "jina_image_chat_template.jinja"


def chat_template_path() -> str:
    """Absolute path to the packaged Jina image chat template (ships in the wheel)."""
    return str(resources.files(__name__) / CHAT_TEMPLATE_FILE)


def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "JinaV4MultiVector",
        "jina_v4_vllm_plugin.model:JinaV4MultiVectorModel",
    )
