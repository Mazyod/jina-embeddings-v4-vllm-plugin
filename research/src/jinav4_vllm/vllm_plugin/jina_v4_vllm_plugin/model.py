"""Jina Embeddings v4 multi-vector model for vLLM (token_embed / late interaction).

Mirrors vLLM's in-tree ColQwen3 pattern, but on the Qwen2.5-VL backbone with Jina's
`multi_vector_projector` (hidden 2048 -> 128, with bias). That projector is NOT present in the
`jinaai/jina-embeddings-v4-vllm-retrieval` checkpoint, so it is injected at load time from an
`.npz` (W [128,2048], b [128]) — default `/artifacts/projector/retrieval.npz`, override with the
`JINA_MV_PROJECTOR` env var.

Serve it natively:
    vllm serve jinaai/jina-embeddings-v4-vllm-retrieval --runner pooling \
        --pooler-config.task token_embed \
        --hf-overrides '{"architectures":["JinaV4MultiVector"]}'
-> /pooling returns final L2-normalized [n,128] multivectors directly.
"""
from __future__ import annotations

import os
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler.tokwise import pooler_for_token_embed
from vllm.model_executor.models.interfaces import SupportsLateInteraction
from vllm.model_executor.models.interfaces_base import default_pooling_type
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLDummyInputsBuilder,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLMultiModalProcessor,
    Qwen2_5_VLProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

PROJECTOR_ENV = "JINA_MV_PROJECTOR"
PROJECTOR_DEFAULT = "/artifacts/projector/retrieval.npz"
MULTIVECTOR_DIM = 128


@default_pooling_type(seq_pooling_type="CLS", tok_pooling_type="ALL")
@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5_VLMultiModalProcessor,
    info=Qwen2_5_VLProcessingInfo,
    dummy_inputs=Qwen2_5_VLDummyInputsBuilder,
)
class JinaV4MultiVectorModel(Qwen2_5_VLForConditionalGeneration, SupportsLateInteraction):
    """Qwen2.5-VL backbone + Jina ColBERT-style projection, producing per-token [n,128]."""

    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None and hasattr(config, "text_config"):
            hidden_size = config.text_config.hidden_size
        # Keep the projector in fp32 and upcast hidden states before projecting, to match the
        # reference projection numerics (the bf16 backbone floor still applies upstream).
        self.multi_vector_projector = nn.Linear(
            hidden_size, MULTIVECTOR_DIM, bias=True, dtype=torch.float32
        )
        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None
        # Projection is applied in forward(); the pooler just gathers all token vectors.
        self.pooler = pooler_for_token_embed(pooler_config, projector=None)

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, **kwargs):
        hidden_states = super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if not isinstance(hidden_states, torch.Tensor):
            return hidden_states
        w = self.multi_vector_projector.weight
        hidden_states = self.multi_vector_projector(hidden_states.to(w.dtype))
        return torch.nn.functional.normalize(hidden_states, p=2, dim=-1)

    def _load_projector(self) -> None:
        path = os.environ.get(PROJECTOR_ENV, PROJECTOR_DEFAULT)
        proj = np.load(path)
        dev = self.multi_vector_projector.weight.device
        with torch.no_grad():
            self.multi_vector_projector.weight.copy_(
                torch.from_numpy(proj["W"]).to(device=dev, dtype=torch.float32))
            self.multi_vector_projector.bias.copy_(
                torch.from_numpy(proj["b"]).to(device=dev, dtype=torch.float32))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Delegate the backbone to the parent so the inherited Qwen2.5-VL hf_to_vllm_mapper applies.
        weights = list(weights)
        proj_weights = {n: w for n, w in weights if "multi_vector_projector" in n}
        backbone = [(n, w) for n, w in weights if "multi_vector_projector" not in n]
        loaded = super().load_weights(backbone)
        loaded = set(loaded) if loaded is not None else set()

        dev = self.multi_vector_projector.weight.device
        if proj_weights:
            # Projector baked into the checkpoint (fully drop-in mode).
            with torch.no_grad():
                for name, w in proj_weights.items():
                    attr = name.rsplit("multi_vector_projector.", 1)[-1]  # "weight" | "bias"
                    getattr(self.multi_vector_projector, attr).copy_(
                        w.to(device=dev, dtype=torch.float32))
                    loaded.add(f"multi_vector_projector.{attr}")
        else:
            # Projector injected from the .npz (JINA_MV_PROJECTOR / default volume path).
            self._load_projector()
            loaded.add("multi_vector_projector.weight")
            loaded.add("multi_vector_projector.bias")
        return loaded
