from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

import torch
from transformers.cache_utils import Cache, CacheLayerMixin

from turboquant.quantization.turboquant_mse import build_scalar_codebook, random_rotation_matrix


def _pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    if indices.ndim != 2:
        raise ValueError(f"Expected packed indices input to have shape [num_vectors, dim], got {indices.shape}.")
    num_vectors, dim = indices.shape
    flat = indices.reshape(-1).to(dtype=torch.int32)
    bit_positions = torch.arange(bits, device=flat.device, dtype=torch.int32)
    value_bits = ((flat.unsqueeze(1) >> bit_positions) & 1).to(dtype=torch.uint8).reshape(-1)
    row_bits = dim * bits
    row_bytes = ceil(row_bits / 8)
    bits_per_row_padded = row_bytes * 8
    if bits_per_row_padded != row_bits:
        value_bits = value_bits.view(num_vectors, row_bits)
        pad_bits = bits_per_row_padded - row_bits
        value_bits = torch.cat(
            (
                value_bits,
                torch.zeros((num_vectors, pad_bits), dtype=torch.uint8, device=value_bits.device),
            ),
            dim=1,
        ).reshape(-1)
    else:
        value_bits = value_bits.view(num_vectors, row_bits).reshape(-1)
    pad_bits = (-int(value_bits.numel())) % 8
    if pad_bits:
        value_bits = torch.cat(
            (value_bits, torch.zeros(pad_bits, dtype=torch.uint8, device=value_bits.device)),
            dim=0,
        )
    byte_weights = (2 ** torch.arange(8, device=value_bits.device, dtype=torch.int32)).to(torch.uint8)
    packed = (value_bits.view(-1, 8) * byte_weights).sum(dim=1).to(dtype=torch.uint8)
    return packed.view(num_vectors, row_bytes).contiguous()


def _unpack_indices(packed: torch.Tensor, bits: int, dim: int) -> torch.Tensor:
    if packed.ndim != 2:
        raise ValueError(f"Expected packed indices tensor with shape [num_vectors, packed_bytes], got {packed.shape}.")
    num_vectors = packed.shape[0]
    byte_positions = torch.arange(8, device=packed.device, dtype=torch.int32)
    bits_flat = ((packed.to(dtype=torch.int32).unsqueeze(-1) >> byte_positions) & 1).reshape(-1)
    bits_flat = bits_flat.view(num_vectors, -1)[:, : dim * bits].reshape(-1)
    value_bits = bits_flat.view(num_vectors * dim, bits)
    bit_weights = 2 ** torch.arange(bits, device=packed.device, dtype=torch.int32)
    return (value_bits * bit_weights).sum(dim=1).to(dtype=torch.int64).view(num_vectors, dim)


@dataclass
class PackedTensorMSE:
    packed_indices: torch.Tensor
    norms: torch.Tensor
    original_shape: tuple[int, int, int, int]
    original_dtype: torch.dtype
    bits: int

    @property
    def num_vectors(self) -> int:
        batch, heads, seq_len, _ = self.original_shape
        return int(batch * heads * seq_len)

    @property
    def dimension(self) -> int:
        return int(self.original_shape[-1])

    def storage_bytes(self) -> int:
        return int(
            self.packed_indices.numel() * self.packed_indices.element_size()
            + self.norms.numel() * self.norms.element_size()
        )

    def append(self, other: "PackedTensorMSE") -> "PackedTensorMSE":
        if self.original_shape[:2] != other.original_shape[:2] or self.dimension != other.dimension:
            raise ValueError("Packed tensor append requires matching batch, head, and dimension shape.")
        batch, heads, seq_len, dim = self.original_shape
        _, _, other_seq_len, _ = other.original_shape
        return PackedTensorMSE(
            packed_indices=torch.cat((self.packed_indices, other.packed_indices), dim=2).contiguous(),
            norms=torch.cat((self.norms, other.norms), dim=2).contiguous(),
            original_shape=(batch, heads, seq_len + other_seq_len, dim),
            original_dtype=self.original_dtype,
            bits=self.bits,
        )


class PackedMSELayer(CacheLayerMixin):
    is_sliding = False

    def __init__(self, bits: int, seed: int = 0, grid_size: int = 32769):
        super().__init__()
        self.bits = bits
        self.seed = seed
        self.grid_size = grid_size
        self.keys_packed: PackedTensorMSE | None = None
        self.values_packed: PackedTensorMSE | None = None
        self.device: torch.device | None = None
        self.dtype: torch.dtype | None = None
        self.rotation: torch.Tensor | None = None
        self.centers: torch.Tensor | None = None
        self.boundaries: torch.Tensor | None = None

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        dimension = int(key_states.shape[-1])
        codebook = build_scalar_codebook(dimension=dimension, bits=self.bits, grid_size=self.grid_size)
        self.rotation = torch.from_numpy(random_rotation_matrix(dimension=dimension, seed=self.seed)).to(
            device=self.device,
            dtype=torch.float32,
        )
        self.centers = torch.from_numpy(codebook.centers.astype("float32")).to(device=self.device)
        self.boundaries = torch.from_numpy(codebook.boundaries.astype("float32")).to(device=self.device)
        self.is_initialized = True

    def initialize_from_dense(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        self.keys_packed = self._encode_tensor(key_states)
        self.values_packed = self._encode_tensor(value_states)

    def _encode_tensor(self, tensor: torch.Tensor) -> PackedTensorMSE:
        if not self.is_initialized:
            self.lazy_initialization(tensor, tensor)
        tensor32 = tensor.detach().to(dtype=torch.float32)
        original_shape = tuple(int(item) for item in tensor32.shape)
        flat = tensor32.reshape(-1, original_shape[-1])
        norms = torch.linalg.norm(flat, dim=1)
        safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
        normalized = flat / safe_norms.unsqueeze(1)
        rotated = normalized @ self.rotation.T
        indices = torch.bucketize(rotated, self.boundaries).to(dtype=torch.int64)
        return PackedTensorMSE(
            packed_indices=_pack_indices(indices, self.bits).view(
                original_shape[0],
                original_shape[1],
                original_shape[2],
                -1,
            ),
            norms=safe_norms.to(dtype=torch.float16).view(
                original_shape[0],
                original_shape[1],
                original_shape[2],
            ),
            original_shape=original_shape,
            original_dtype=tensor.dtype,
            bits=self.bits,
        )

    def _decode_tensor(self, packed: PackedTensorMSE) -> torch.Tensor:
        indices = _unpack_indices(
            packed.packed_indices.view(packed.num_vectors, -1),
            packed.bits,
            packed.dimension,
        )
        rotated = self.centers[indices]
        reconstructed = (rotated @ self.rotation) * packed.norms.view(-1).to(dtype=torch.float32).unsqueeze(1)
        return reconstructed.view(packed.original_shape).to(dtype=packed.original_dtype)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        previous_keys = (
            self._decode_tensor(self.keys_packed)
            if self.keys_packed is not None
            else key_states.new_empty((*key_states.shape[:2], 0, key_states.shape[-1]))
        )
        previous_values = (
            self._decode_tensor(self.values_packed)
            if self.values_packed is not None
            else value_states.new_empty((*value_states.shape[:2], 0, value_states.shape[-1]))
        )
        new_keys_packed = self._encode_tensor(key_states)
        new_values_packed = self._encode_tensor(value_states)
        self.keys_packed = (
            new_keys_packed
            if self.keys_packed is None
            else self.keys_packed.append(new_keys_packed)
        )
        self.values_packed = (
            new_values_packed
            if self.values_packed is None
            else self.values_packed.append(new_values_packed)
        )
        full_key_states = torch.cat((previous_keys, self._decode_tensor(new_keys_packed)), dim=-2)
        full_value_states = torch.cat((previous_values, self._decode_tensor(new_values_packed)), dim=-2)
        return full_key_states, full_value_states

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        kv_offset = 0
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        if self.keys_packed is None:
            return 0
        return int(self.keys_packed.original_shape[-2])

    def get_max_cache_shape(self) -> int:
        return -1

    def storage_bytes(self) -> int:
        key_bytes = self.keys_packed.storage_bytes() if self.keys_packed is not None else 0
        value_bytes = self.values_packed.storage_bytes() if self.values_packed is not None else 0
        return int(key_bytes + value_bytes)


def build_packed_mse_cache(
    past_key_values,
    bits: int,
    seed: int = 0,
    grid_size: int = 32769,
) -> Cache:
    layers = []
    for key_states, value_states, *_ in past_key_values:
        layer = PackedMSELayer(bits=bits, seed=seed, grid_size=grid_size)
        layer.initialize_from_dense(key_states, value_states)
        layers.append(layer)
    return Cache(layers=layers)


def packed_cache_storage_bytes(cache: Cache) -> int:
    total = 0
    for layer in cache.layers:
        if hasattr(layer, "storage_bytes"):
            total += int(layer.storage_bytes())
    return total


def packed_cache_storage_breakdown(cache: Cache) -> dict[str, int]:
    layer_bytes = [int(layer.storage_bytes()) for layer in cache.layers if hasattr(layer, "storage_bytes")]
    return {
        "packed_total_bytes": int(sum(layer_bytes)),
        "packed_layer_mean_bytes": int(sum(layer_bytes) / len(layer_bytes)) if layer_bytes else 0,
        "packed_num_layers": len(layer_bytes),
    }
