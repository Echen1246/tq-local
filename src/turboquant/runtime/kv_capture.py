from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


def model_config_summary(model) -> dict[str, Any]:
    config = model.config
    return {
        "model_type": getattr(config, "model_type", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "max_position_embeddings": getattr(config, "max_position_embeddings", None),
        "sliding_window": getattr(config, "sliding_window", None),
        "rope_scaling": getattr(config, "rope_scaling", None),
    }


def _legacy_past_key_values(past_key_values):
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    return past_key_values


def _extract_key_value_tensors(layer_cache) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(layer_cache, (tuple, list)):
        raise TypeError(f"Unsupported layer cache type: {type(layer_cache)!r}")

    tensor_items = [item for item in layer_cache if torch.is_tensor(item)]
    if len(tensor_items) < 2:
        raise ValueError(
            f"Expected at least two tensor entries in a layer cache, got {len(tensor_items)}."
        )
    return tensor_items[0], tensor_items[1]


def summarize_past_key_values(past_key_values) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for layer_index, layer_cache in enumerate(_legacy_past_key_values(past_key_values)):
        key, value = _extract_key_value_tensors(layer_cache)
        key_cpu = key.detach().to("cpu")
        value_cpu = value.detach().to("cpu")
        key_token_norms = key_cpu.float().norm(dim=-1)
        value_token_norms = value_cpu.float().norm(dim=-1)
        summaries.append(
            {
                "layer": layer_index,
                "layer_cache_entries": len(layer_cache) if isinstance(layer_cache, (tuple, list)) else None,
                "key_shape": list(key_cpu.shape),
                "value_shape": list(value_cpu.shape),
                "dtype": str(key_cpu.dtype),
                "key_mean_token_norm": float(key_token_norms.mean().item()),
                "key_std_token_norm": float(key_token_norms.std().item()),
                "value_mean_token_norm": float(value_token_norms.mean().item()),
                "value_std_token_norm": float(value_token_norms.std().item()),
            }
        )
    return summaries


def past_key_values_to_state_dict(past_key_values) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    for layer_index, layer_cache in enumerate(_legacy_past_key_values(past_key_values)):
        key, value = _extract_key_value_tensors(layer_cache)
        state_dict[f"layer_{layer_index:02d}.key"] = key.detach().to("cpu").contiguous()
        state_dict[f"layer_{layer_index:02d}.value"] = value.detach().to("cpu").contiguous()
    return state_dict


def save_past_key_values(path: Path, past_key_values) -> None:
    save_file(past_key_values_to_state_dict(past_key_values), str(path))
