from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


def load_kv_artifact(path: str | Path) -> dict[str, object]:
    tensor_map = load_file(str(path))
    return {name: tensor.cpu() for name, tensor in tensor_map.items()}


def tensor_map_layers(tensor_map: dict[str, object]) -> list[int]:
    layers: set[int] = set()
    for name in tensor_map:
        prefix, _ = name.split(".", maxsplit=1)
        _, layer_index = prefix.split("_", maxsplit=1)
        layers.add(int(layer_index))
    return sorted(layers)


def extract_layer_tensor_array(
    tensor_map: dict[str, object],
    layer_index: int,
    kind: str,
) -> np.ndarray:
    tensor = tensor_map[f"layer_{layer_index:02d}.{kind}"]
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(dtype=torch.float32)
    return tensor.numpy()


def extract_layer_vectors(
    tensor_map: dict[str, object],
    layer_index: int,
    kind: str,
) -> np.ndarray:
    if kind not in {"key", "value"}:
        raise ValueError(f"Unsupported kind={kind!r}")
    array = extract_layer_tensor_array(tensor_map, layer_index=layer_index, kind=kind)
    if array.ndim != 4:
        raise ValueError(
            f"Expected KV tensor to have 4 dims [batch, kv_heads, seq, head_dim], got {array.shape}."
        )
    batch, kv_heads, seq_len, head_dim = array.shape
    return array.reshape(batch * kv_heads * seq_len, head_dim)
