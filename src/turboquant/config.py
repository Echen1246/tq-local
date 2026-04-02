from __future__ import annotations

MODEL_ID = "Qwen/QwQ-32B"
PINNED_MODEL_REVISION = "976055f8c83f394f35dbd3ab09a285a984907bd0"

MODAL_APP_NAME = "turboquant-qwq-baseline"
GPU_TYPE = "H200"
CPU_CORES = 4
MEMORY_MB = 16384

HF_CACHE_DIR = "/vol/hf-cache"
ARTIFACTS_DIR = "/vol/artifacts"

DEFAULT_ATTN_IMPLEMENTATION = "sdpa"
SUPPORTED_ATTN_IMPLEMENTATIONS = ("sdpa", "eager", "flash_attention_2")
DEFAULT_MAX_NEW_TOKENS = 256


def resolve_revision(revision: str | None) -> str:
    return revision or PINNED_MODEL_REVISION
