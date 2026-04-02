from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from turboquant.config import DEFAULT_ATTN_IMPLEMENTATION, MODEL_ID, PINNED_MODEL_REVISION

@dataclass(frozen=True)
class QwQLoadConfig:
    model_id: str = MODEL_ID
    revision: str | None = PINNED_MODEL_REVISION
    dtype: torch.dtype = torch.bfloat16
    attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION
    device_map: str = "auto"
    trust_remote_code: bool = False
    token: str | None = None
    cache_dir: str | None = None


def load_qwq_model(config: QwQLoadConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.revision,
        token=config.token,
        cache_dir=config.cache_dir,
        trust_remote_code=config.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        revision=config.revision,
        token=config.token,
        cache_dir=config.cache_dir,
        dtype=config.dtype,
        device_map=config.device_map,
        attn_implementation=config.attn_implementation,
        low_cpu_mem_usage=True,
        trust_remote_code=config.trust_remote_code,
    )
    model.eval()
    return tokenizer, model
