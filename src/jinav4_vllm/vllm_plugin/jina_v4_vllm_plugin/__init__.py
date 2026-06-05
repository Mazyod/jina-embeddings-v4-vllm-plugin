"""vLLM general plugin: register the JinaV4MultiVector architecture in every vLLM process."""


def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "JinaV4MultiVector",
        "jina_v4_vllm_plugin.model:JinaV4MultiVectorModel",
    )
