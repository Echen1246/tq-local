from __future__ import annotations

from math import ceil
from typing import Any

import torch


def _layer_tensor_entries(layer_cache) -> list[torch.Tensor]:
    if isinstance(layer_cache, (tuple, list)):
        return [item for item in layer_cache if torch.is_tensor(item)]

    tensors: list[torch.Tensor] = []
    if getattr(layer_cache, "keys", None) is not None:
        tensors.append(layer_cache.keys)
    if getattr(layer_cache, "values", None) is not None:
        tensors.append(layer_cache.values)
    sliding_tensor = getattr(layer_cache, "_sliding_window_tensor", None)
    if torch.is_tensor(sliding_tensor):
        tensors.append(sliding_tensor)
    return tensors


def _iter_layer_caches(past_key_values):
    if hasattr(past_key_values, "layers"):
        return past_key_values.layers
    return past_key_values


def _tensor_num_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def past_key_values_memory_breakdown(past_key_values) -> dict[str, Any]:
    key_bytes = 0
    value_bytes = 0
    extra_tensor_bytes = 0
    num_layers = 0
    num_key_value_vectors = 0
    vector_dimension = None

    for layer_cache in _iter_layer_caches(past_key_values):
        tensors = _layer_tensor_entries(layer_cache)
        if len(tensors) < 2:
            continue
        key, value, *extras = tensors
        key_bytes += _tensor_num_bytes(key)
        value_bytes += _tensor_num_bytes(value)
        extra_tensor_bytes += sum(_tensor_num_bytes(item) for item in extras)
        num_layers += 1
        if vector_dimension is None:
            vector_dimension = int(key.shape[-1])
        num_key_value_vectors += int(key.shape[0] * key.shape[1] * key.shape[2])

    return {
        "num_layers": num_layers,
        "vector_dimension": vector_dimension,
        "num_key_value_vectors_per_kind": num_key_value_vectors,
        "key_bytes": key_bytes,
        "value_bytes": value_bytes,
        "dense_kv_bytes": key_bytes + value_bytes,
        "extra_tensor_bytes": extra_tensor_bytes,
        "total_cache_tensor_bytes": key_bytes + value_bytes + extra_tensor_bytes,
    }


def turboquant_mse_packed_bytes(
    *,
    num_vectors_per_kind: int,
    vector_dimension: int,
    bits: int,
    norm_bytes: int = 2,
) -> dict[str, int | str]:
    per_kind_index_bits = num_vectors_per_kind * vector_dimension * bits
    per_kind_index_bytes = ceil(per_kind_index_bits / 8)
    per_kind_norm_bytes = num_vectors_per_kind * norm_bytes
    packed_bytes = 2 * (per_kind_index_bytes + per_kind_norm_bytes)
    return {
        "bits": bits,
        "norm_bytes_per_vector": norm_bytes,
        "norm_dtype_assumption": "float16" if norm_bytes == 2 else f"{norm_bytes * 8}-bit",
        "packed_kv_bytes": packed_bytes,
        "packed_index_bytes": 2 * per_kind_index_bytes,
        "packed_norm_bytes": 2 * per_kind_norm_bytes,
    }


def gpu_peak_memory_bytes() -> dict[str, int | None]:
    if not torch.cuda.is_available():
        return {
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def gpu_current_memory_bytes() -> dict[str, int | None]:
    if not torch.cuda.is_available():
        return {
            "allocated_bytes": None,
            "reserved_bytes": None,
        }
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
    }
