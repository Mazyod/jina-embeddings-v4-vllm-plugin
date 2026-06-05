# Jina v4 multi-vector on the OFFICIAL vLLM OpenAI server image (Variant C).
#
# Extends `vllm/vllm-openai` with the JinaV4MultiVector out-of-tree model plugin so the stock
# OpenAI server's /pooling endpoint returns final L2-normalized [n,128] multivectors for text+image.
#
# Pin VLLM_TAG to the version the plugin was validated against (the model class touches vLLM
# internals; re-validate on upgrades — see reports/variant_c.md).
ARG VLLM_TAG=v0.22.0
FROM vllm/vllm-openai:${VLLM_TAG}

# 1) Install the plugin (registers the JinaV4MultiVector architecture via a vllm.general_plugins
#    entry point, loaded in every process incl. the v1 EngineCore worker).
COPY src/jinav4_vllm/vllm_plugin /opt/jina_plugin
RUN python3 -m pip install --no-cache-dir --no-deps /opt/jina_plugin

# 2) Chat template (only needed for the "Mode A" run below; the baked checkpoint embeds its own).
#    Keep a copy at a stable path.
RUN cp /opt/jina_plugin/jina_image_chat_template.jinja /opt/jina_image_chat_template.jinja

# 3) (Mode A only) the projector, if you are NOT using the baked checkpoint. Uncomment + provide it:
# COPY artifacts/projector/retrieval.npz /opt/retrieval.npz
# ENV JINA_MV_PROJECTOR=/opt/retrieval.npz

# The base image's entrypoint is already `vllm serve`. Pass the model + flags at `docker run` time
# (see deploy/DEPLOY.md). Example (Mode B, fully drop-in baked checkpoint):
#   docker run --gpus all -p 8000:8000 <image> \
#     <your-baked-checkpoint-repo-or-path> \
#     --runner pooling --pooler-config.task token_embed
