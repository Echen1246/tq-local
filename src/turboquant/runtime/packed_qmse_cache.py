from __future__ import annotations

from dataclasses import dataclass
from math import ceil, pi, sqrt
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


def detect_outlier_channels(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    num_outlier_channels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Identify outlier channels by combined key+value L2 energy.

    Returns ``(outlier_indices, normal_indices)``, both sorted.
    """
    key_energy = key_states.float().pow(2).sum(dim=(0, 1, 2))
    value_energy = value_states.float().pow(2).sum(dim=(0, 1, 2))
    combined = key_energy + value_energy
    _, top_indices = torch.topk(combined, num_outlier_channels)
    outlier_indices = top_indices.sort().values

    full_dim = key_states.shape[-1]
    mask = torch.ones(full_dim, dtype=torch.bool, device=key_states.device)
    mask[outlier_indices] = False
    normal_indices = torch.arange(full_dim, device=key_states.device)[mask]
    return outlier_indices, normal_indices


_OUTLIER_SEED_OFFSET = 7919
_QJL_SEED_OFFSET = 31337


@dataclass
class PackedQJL:
    """1-bit QJL sign vectors and per-vector residual norms."""

    packed_signs: torch.Tensor
    residual_norms: torch.Tensor
    dimension: int

    def storage_bytes(self) -> int:
        return int(
            self.packed_signs.numel() * self.packed_signs.element_size()
            + self.residual_norms.numel() * self.residual_norms.element_size()
        )

    def append(self, other: "PackedQJL") -> "PackedQJL":
        return PackedQJL(
            packed_signs=torch.cat((self.packed_signs, other.packed_signs), dim=2).contiguous(),
            residual_norms=torch.cat((self.residual_norms, other.residual_norms), dim=2).contiguous(),
            dimension=self.dimension,
        )


def _generate_qjl_matrix(dimension: int, seed: int) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    return torch.randn(dimension, dimension, generator=rng, dtype=torch.float32)


def _pack_qjl_signs(signs: torch.Tensor, original_shape: tuple[int, ...]) -> PackedQJL:
    """Encode {-1,+1} sign tensor of shape [batch, heads, seq, dim]."""
    batch, heads, seq_len, dim = original_shape
    flat = signs.reshape(-1, dim)
    sign_bits = ((flat + 1) // 2).to(dtype=torch.int64)
    packed = _pack_indices(sign_bits, bits=1).view(batch, heads, seq_len, -1)
    return PackedQJL(packed_signs=packed, residual_norms=torch.empty(0), dimension=dim)


def _unpack_qjl_signs(packed: PackedQJL, num_vectors: int) -> torch.Tensor:
    flat = packed.packed_signs.view(num_vectors, -1)
    bits = _unpack_indices(flat, bits=1, dim=packed.dimension)
    return (2 * bits.to(torch.float32) - 1)


class PackedMSELayer(CacheLayerMixin):
    is_sliding = False

    def __init__(
        self,
        bits: int,
        seed: int = 0,
        grid_size: int = 32769,
        num_outlier_channels: int = 0,
        outlier_extra_bits: int = 1,
        use_qjl_keys: bool = False,
        quantize_decode: bool = False,
    ):
        super().__init__()
        self.bits = bits
        self.seed = seed
        self.grid_size = grid_size
        self.num_outlier_channels = num_outlier_channels
        self.outlier_extra_bits = outlier_extra_bits
        self.use_qjl_keys = use_qjl_keys
        self.quantize_decode = quantize_decode

        self.keys_packed: PackedTensorMSE | None = None
        self.values_packed: PackedTensorMSE | None = None
        self.device: torch.device | None = None
        self.dtype: torch.dtype | None = None
        self.rotation: torch.Tensor | None = None
        self.centers: torch.Tensor | None = None
        self.boundaries: torch.Tensor | None = None

        self._outlier_indices: torch.Tensor | None = None
        self._normal_indices: torch.Tensor | None = None
        self._outlier_rotation: torch.Tensor | None = None
        self._outlier_centers: torch.Tensor | None = None
        self._outlier_boundaries: torch.Tensor | None = None
        self._keys_outlier: PackedTensorMSE | None = None
        self._values_outlier: PackedTensorMSE | None = None

        # QJL state for keys (Q_prod): keys use MSE at bits-1 + 1-bit QJL
        self._qjl_matrix: torch.Tensor | None = None
        self._key_rotation: torch.Tensor | None = None
        self._key_centers: torch.Tensor | None = None
        self._key_boundaries: torch.Tensor | None = None
        self._keys_qjl: PackedQJL | None = None
        # Outlier-path QJL state
        self._qjl_outlier_matrix: torch.Tensor | None = None
        self._key_outlier_rotation: torch.Tensor | None = None
        self._key_outlier_centers: torch.Tensor | None = None
        self._key_outlier_boundaries: torch.Tensor | None = None
        self._keys_outlier_qjl: PackedQJL | None = None

        self._force_dense: bool = False
        self._lazy_update: bool = False

        # Dense buffer for generated tokens (when quantize_decode=False)
        self._dense_keys: torch.Tensor | None = None
        self._dense_values: torch.Tensor | None = None

    @property
    def _outlier_enabled(self) -> bool:
        return self._outlier_indices is not None

    def _build_codebook_tensors(
        self, dimension: int, bits: int, seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cb = build_scalar_codebook(dimension=dimension, bits=bits, grid_size=self.grid_size)
        rotation = torch.from_numpy(
            random_rotation_matrix(dimension=dimension, seed=seed),
        ).to(device=self.device, dtype=torch.float32)
        centers = torch.from_numpy(cb.centers.astype("float32")).to(device=self.device)
        boundaries = torch.from_numpy(cb.boundaries.astype("float32")).to(device=self.device)
        return rotation, centers, boundaries

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        full_dim = int(key_states.shape[-1])
        key_mse_bits = (self.bits - 1) if self.use_qjl_keys else self.bits

        if self.num_outlier_channels > 0 and self.num_outlier_channels < full_dim:
            self._outlier_indices, self._normal_indices = detect_outlier_channels(
                key_states, value_states, self.num_outlier_channels,
            )
            normal_dim = int(self._normal_indices.shape[0])
            outlier_dim = int(self._outlier_indices.shape[0])
            outlier_bits = self.bits + self.outlier_extra_bits
            key_outlier_mse_bits = (outlier_bits - 1) if self.use_qjl_keys else outlier_bits

            self.rotation, self.centers, self.boundaries = self._build_codebook_tensors(
                normal_dim, self.bits, self.seed,
            )
            outlier_seed = self.seed + _OUTLIER_SEED_OFFSET
            self._outlier_rotation, self._outlier_centers, self._outlier_boundaries = (
                self._build_codebook_tensors(outlier_dim, outlier_bits, outlier_seed)
            )

            if self.use_qjl_keys:
                self._key_rotation, self._key_centers, self._key_boundaries = (
                    self._build_codebook_tensors(normal_dim, key_mse_bits, self.seed + 1)
                )
                self._key_outlier_rotation, self._key_outlier_centers, self._key_outlier_boundaries = (
                    self._build_codebook_tensors(outlier_dim, key_outlier_mse_bits, outlier_seed + 1)
                )
                qjl_seed = self.seed + _QJL_SEED_OFFSET
                self._qjl_matrix = _generate_qjl_matrix(normal_dim, qjl_seed).to(device=self.device)
                self._qjl_outlier_matrix = _generate_qjl_matrix(
                    outlier_dim, qjl_seed + 1,
                ).to(device=self.device)
        else:
            self.rotation, self.centers, self.boundaries = self._build_codebook_tensors(
                full_dim, self.bits, self.seed,
            )
            if self.use_qjl_keys:
                self._key_rotation, self._key_centers, self._key_boundaries = (
                    self._build_codebook_tensors(full_dim, key_mse_bits, self.seed + 1)
                )
                qjl_seed = self.seed + _QJL_SEED_OFFSET
                self._qjl_matrix = _generate_qjl_matrix(full_dim, qjl_seed).to(device=self.device)

        self.is_initialized = True

    def _encode_keys_qjl_group(
        self,
        key_states: torch.Tensor,
        rotation: torch.Tensor,
        centers: torch.Tensor,
        boundaries: torch.Tensor,
        bits: int,
        qjl_matrix: torch.Tensor,
    ) -> tuple[PackedTensorMSE, PackedQJL]:
        """Encode keys with Q_prod: MSE at (bits) + 1-bit QJL on residual."""
        mse_packed = self._encode_group(key_states, rotation, centers, boundaries, bits)
        mse_decoded = self._decode_group(mse_packed, rotation, centers)
        residual = (key_states.float() - mse_decoded.float())
        original_shape = tuple(int(x) for x in key_states.shape)
        flat_residual = residual.reshape(-1, original_shape[-1])
        residual_norms = torch.linalg.norm(flat_residual, dim=1)
        safe_norms = torch.where(residual_norms > 0, residual_norms, torch.ones_like(residual_norms))
        normalized_residual = flat_residual / safe_norms.unsqueeze(1)
        projected = normalized_residual @ qjl_matrix.T
        signs = torch.sign(projected)
        signs[signs == 0] = 1.0
        qjl_packed = _pack_qjl_signs(signs.view(original_shape), original_shape)
        qjl_packed.residual_norms = safe_norms.to(dtype=torch.float16).view(
            original_shape[0], original_shape[1], original_shape[2],
        )
        return mse_packed, qjl_packed

    def _decode_keys_qjl_group(
        self,
        mse_packed: PackedTensorMSE,
        qjl: PackedQJL,
        rotation: torch.Tensor,
        centers: torch.Tensor,
        qjl_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Decode Q_prod keys: MSE reconstruction + QJL correction."""
        mse_decoded = self._decode_group(mse_packed, rotation, centers)
        dim = qjl.dimension
        num_vectors = mse_packed.num_vectors
        signs = _unpack_qjl_signs(qjl, num_vectors)
        correction = signs @ qjl_matrix
        scale = sqrt(pi / 2) / dim
        norms = qjl.residual_norms.view(-1).to(dtype=torch.float32)
        correction = (correction * norms.unsqueeze(1) * scale)
        correction = correction.view(mse_packed.original_shape).to(dtype=mse_decoded.dtype)
        return mse_decoded + correction

    def initialize_from_dense(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *,
        force_dense: bool = False,
    ) -> None:
        if force_dense:
            self.device = key_states.device
            self.dtype = key_states.dtype
            self._dense_keys = key_states
            self._dense_values = value_states
            self._force_dense = True
            self.is_initialized = True
            return

        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        if self._outlier_enabled:
            if self.use_qjl_keys:
                self.keys_packed, self._keys_qjl, self._keys_outlier, self._keys_outlier_qjl = (
                    self._encode_keys_split_qjl(key_states)
                )
            else:
                self.keys_packed, self._keys_outlier = self._encode_split(key_states)
            self.values_packed, self._values_outlier = self._encode_split(value_states)
        elif self.use_qjl_keys:
            self.keys_packed, self._keys_qjl = self._encode_keys_qjl_group(
                key_states,
                self._key_rotation, self._key_centers, self._key_boundaries,
                self.bits - 1, self._qjl_matrix,
            )
            self.values_packed = self._encode_group(
                value_states, self.rotation, self.centers, self.boundaries, self.bits,
            )
        else:
            self.keys_packed = self._encode_group(
                key_states, self.rotation, self.centers, self.boundaries, self.bits,
            )
            self.values_packed = self._encode_group(
                value_states, self.rotation, self.centers, self.boundaries, self.bits,
            )

    # ------------------------------------------------------------------
    # Low-level encode / decode for a single channel group
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_group(
        tensor: torch.Tensor,
        rotation: torch.Tensor,
        centers: torch.Tensor,
        boundaries: torch.Tensor,
        bits: int,
    ) -> PackedTensorMSE:
        tensor32 = tensor.detach().to(dtype=torch.float32)
        original_shape = tuple(int(x) for x in tensor32.shape)
        flat = tensor32.reshape(-1, original_shape[-1])
        norms = torch.linalg.norm(flat, dim=1)
        safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
        normalized = flat / safe_norms.unsqueeze(1)
        rotated = normalized @ rotation.T
        indices = torch.bucketize(rotated, boundaries).to(dtype=torch.int64)
        return PackedTensorMSE(
            packed_indices=_pack_indices(indices, bits).view(
                original_shape[0], original_shape[1], original_shape[2], -1,
            ),
            norms=safe_norms.to(dtype=torch.float16).view(
                original_shape[0], original_shape[1], original_shape[2],
            ),
            original_shape=original_shape,
            original_dtype=tensor.dtype,
            bits=bits,
        )

    @staticmethod
    def _decode_group(
        packed: PackedTensorMSE,
        rotation: torch.Tensor,
        centers: torch.Tensor,
    ) -> torch.Tensor:
        if packed.packed_indices.is_cuda:
            try:
                from turboquant.runtime.triton_kernels import triton_available, triton_decode_group

                if triton_available():
                    return triton_decode_group(
                        packed.packed_indices.view(packed.num_vectors, -1),
                        packed.norms,
                        rotation,
                        centers,
                        packed.bits,
                        packed.dimension,
                        packed.original_shape,
                        packed.original_dtype,
                    )
            except Exception:
                pass

        indices = _unpack_indices(
            packed.packed_indices.view(packed.num_vectors, -1),
            packed.bits,
            packed.dimension,
        )
        rotated = centers[indices]
        reconstructed = (rotated @ rotation) * packed.norms.view(-1).to(dtype=torch.float32).unsqueeze(1)
        return reconstructed.view(packed.original_shape).to(dtype=packed.original_dtype)

    # ------------------------------------------------------------------
    # Outlier-aware split encode / merge decode
    # ------------------------------------------------------------------

    def _encode_keys_split_qjl(
        self, tensor: torch.Tensor,
    ) -> tuple[PackedTensorMSE, PackedQJL, PackedTensorMSE, PackedQJL]:
        """Encode keys with outlier split AND QJL. Returns (normal_mse, normal_qjl, outlier_mse, outlier_qjl)."""
        normal_tensor = tensor[..., self._normal_indices]
        outlier_tensor = tensor[..., self._outlier_indices]
        outlier_bits = self.bits + self.outlier_extra_bits
        normal_mse, normal_qjl = self._encode_keys_qjl_group(
            normal_tensor,
            self._key_rotation, self._key_centers, self._key_boundaries,
            self.bits - 1, self._qjl_matrix,
        )
        outlier_mse, outlier_qjl = self._encode_keys_qjl_group(
            outlier_tensor,
            self._key_outlier_rotation, self._key_outlier_centers, self._key_outlier_boundaries,
            outlier_bits - 1, self._qjl_outlier_matrix,
        )
        return normal_mse, normal_qjl, outlier_mse, outlier_qjl

    def _decode_keys_merge_qjl(
        self,
        normal_mse: PackedTensorMSE,
        normal_qjl: PackedQJL,
        outlier_mse: PackedTensorMSE,
        outlier_qjl: PackedQJL,
    ) -> torch.Tensor:
        normal_decoded = self._decode_keys_qjl_group(
            normal_mse, normal_qjl,
            self._key_rotation, self._key_centers, self._qjl_matrix,
        )
        outlier_decoded = self._decode_keys_qjl_group(
            outlier_mse, outlier_qjl,
            self._key_outlier_rotation, self._key_outlier_centers, self._qjl_outlier_matrix,
        )
        batch, heads, seq_len = normal_decoded.shape[:3]
        full_dim = int(self._normal_indices.shape[0] + self._outlier_indices.shape[0])
        full = torch.zeros(
            batch, heads, seq_len, full_dim,
            dtype=normal_decoded.dtype, device=normal_decoded.device,
        )
        full[..., self._normal_indices] = normal_decoded
        full[..., self._outlier_indices] = outlier_decoded
        return full

    def _encode_split(
        self, tensor: torch.Tensor,
    ) -> tuple[PackedTensorMSE, PackedTensorMSE]:
        normal_tensor = tensor[..., self._normal_indices]
        outlier_tensor = tensor[..., self._outlier_indices]
        normal_packed = self._encode_group(
            normal_tensor, self.rotation, self.centers, self.boundaries, self.bits,
        )
        outlier_packed = self._encode_group(
            outlier_tensor,
            self._outlier_rotation,
            self._outlier_centers,
            self._outlier_boundaries,
            self.bits + self.outlier_extra_bits,
        )
        return normal_packed, outlier_packed

    def _decode_merge(
        self,
        normal_packed: PackedTensorMSE,
        outlier_packed: PackedTensorMSE,
    ) -> torch.Tensor:
        normal_decoded = self._decode_group(normal_packed, self.rotation, self.centers)
        outlier_decoded = self._decode_group(
            outlier_packed, self._outlier_rotation, self._outlier_centers,
        )
        batch, heads, seq_len = normal_decoded.shape[:3]
        full_dim = int(self._normal_indices.shape[0] + self._outlier_indices.shape[0])
        full = torch.zeros(
            batch, heads, seq_len, full_dim,
            dtype=normal_decoded.dtype, device=normal_decoded.device,
        )
        full[..., self._normal_indices] = normal_decoded
        full[..., self._outlier_indices] = outlier_decoded
        return full

    # ------------------------------------------------------------------
    # Cache protocol: update
    # ------------------------------------------------------------------

    def _decode_keys_full(self) -> torch.Tensor | None:
        """Decode all stored key vectors, dispatching to the correct path."""
        if self.keys_packed is None:
            return None
        if self._outlier_enabled and self.use_qjl_keys:
            return self._decode_keys_merge_qjl(
                self.keys_packed, self._keys_qjl,
                self._keys_outlier, self._keys_outlier_qjl,
            )
        if self._outlier_enabled:
            return self._decode_merge(self.keys_packed, self._keys_outlier)
        if self.use_qjl_keys:
            return self._decode_keys_qjl_group(
                self.keys_packed, self._keys_qjl,
                self._key_rotation, self._key_centers, self._qjl_matrix,
            )
        return self._decode_group(self.keys_packed, self.rotation, self.centers)

    def _decode_values_full(self) -> torch.Tensor | None:
        """Decode all stored value vectors, dispatching to the correct path."""
        if self.values_packed is None:
            return None
        if self._outlier_enabled:
            return self._decode_merge(self.values_packed, self._values_outlier)
        return self._decode_group(self.values_packed, self.rotation, self.centers)

    # ------------------------------------------------------------------
    # Range decoding for chunked attention
    # ------------------------------------------------------------------

    @staticmethod
    def _slice_packed(packed: PackedTensorMSE, start: int, end: int) -> PackedTensorMSE:
        """Slice a PackedTensorMSE along the sequence dimension."""
        b, h, s, d = packed.original_shape
        return PackedTensorMSE(
            packed_indices=packed.packed_indices[:, :, start:end, :].contiguous(),
            norms=packed.norms[:, :, start:end].contiguous(),
            original_shape=(b, h, end - start, d),
            original_dtype=packed.original_dtype,
            bits=packed.bits,
        )

    @staticmethod
    def _slice_qjl(qjl: PackedQJL, start: int, end: int, batch: int, heads: int) -> PackedQJL:
        """Slice a PackedQJL along the sequence dimension."""
        return PackedQJL(
            packed_signs=qjl.packed_signs[:, :, start:end, :].contiguous(),
            residual_norms=qjl.residual_norms[:, :, start:end].contiguous(),
            dimension=qjl.dimension,
        )

    def _decode_keys_range(self, start: int, end: int) -> torch.Tensor:
        """Decode keys for positions [start:end] only."""
        kp_slice = self._slice_packed(self.keys_packed, start, end)
        if self._outlier_enabled and self.use_qjl_keys:
            ko_slice = self._slice_packed(self._keys_outlier, start, end)
            kq_slice = self._slice_qjl(self._keys_qjl, start, end,
                                        *self.keys_packed.original_shape[:2])
            koq_slice = self._slice_qjl(self._keys_outlier_qjl, start, end,
                                         *self._keys_outlier.original_shape[:2])
            return self._decode_keys_merge_qjl(kp_slice, kq_slice, ko_slice, koq_slice)
        if self._outlier_enabled:
            ko_slice = self._slice_packed(self._keys_outlier, start, end)
            return self._decode_merge(kp_slice, ko_slice)
        if self.use_qjl_keys:
            kq_slice = self._slice_qjl(self._keys_qjl, start, end,
                                        *self.keys_packed.original_shape[:2])
            return self._decode_keys_qjl_group(
                kp_slice, kq_slice,
                self._key_rotation, self._key_centers, self._qjl_matrix,
            )
        return self._decode_group(kp_slice, self.rotation, self.centers)

    def _decode_values_range(self, start: int, end: int) -> torch.Tensor:
        """Decode values for positions [start:end] only."""
        vp_slice = self._slice_packed(self.values_packed, start, end)
        if self._outlier_enabled:
            vo_slice = self._slice_packed(self._values_outlier, start, end)
            return self._decode_merge(vp_slice, vo_slice)
        return self._decode_group(vp_slice, self.rotation, self.centers)

    def packed_seq_length(self) -> int:
        """Number of tokens in packed (compressed) storage."""
        if self.keys_packed is None:
            return 0
        return int(self.keys_packed.original_shape[-2])

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        if self._lazy_update and not self._force_dense:
            return self._update_lazy(key_states, value_states)

        if self._force_dense:
            return self._update_dense_decode(key_states, value_states)

        if not self.quantize_decode:
            return self._update_dense_decode(key_states, value_states)

        if self._outlier_enabled and self.use_qjl_keys:
            return self._update_split_qjl(key_states, value_states)
        if self._outlier_enabled:
            return self._update_split(key_states, value_states)
        if self.use_qjl_keys:
            return self._update_flat_qjl(key_states, value_states)
        return self._update_flat(key_states, value_states)

    def _update_lazy(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Store the new token without decompressing history.

        Used when the turboquant attention backend handles compressed KV
        directly.  Only appends to the dense decode buffer (or encodes
        into packed storage when ``quantize_decode`` is set); returns
        just the new token's K/V so the custom attention function can
        combine them with the compressed history itself.
        """
        if self.quantize_decode:
            if self.use_qjl_keys:
                new_k_mse, new_k_qjl = self._encode_keys_qjl_group(
                    key_states,
                    self._key_rotation, self._key_centers, self._key_boundaries,
                    self.bits - 1, self._qjl_matrix,
                )
                self.keys_packed = (
                    new_k_mse if self.keys_packed is None
                    else self.keys_packed.append(new_k_mse)
                )
                self._keys_qjl = (
                    new_k_qjl if self._keys_qjl is None
                    else self._keys_qjl.append(new_k_qjl)
                )
            else:
                new_kp = self._encode_group(
                    key_states, self.rotation, self.centers, self.boundaries, self.bits,
                )
                self.keys_packed = (
                    new_kp if self.keys_packed is None
                    else self.keys_packed.append(new_kp)
                )
            new_vp = self._encode_group(
                value_states, self.rotation, self.centers, self.boundaries, self.bits,
            )
            self.values_packed = (
                new_vp if self.values_packed is None
                else self.values_packed.append(new_vp)
            )
        else:
            if self._dense_keys is None:
                self._dense_keys = key_states
            else:
                self._dense_keys = torch.cat((self._dense_keys, key_states), dim=-2)
            if self._dense_values is None:
                self._dense_values = value_states
            else:
                self._dense_values = torch.cat((self._dense_values, value_states), dim=-2)

        return key_states, value_states

    def _update_dense_decode(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep generated tokens in a dense buffer instead of re-quantizing.

        Only the prefill cache is compressed; decode tokens stay full-precision.
        This eliminates compounding quantization noise during autoregressive
        generation while preserving memory savings from the (much larger)
        prefill cache.
        """
        packed_keys = self._decode_keys_full()
        packed_values = self._decode_values_full()
        if self._dense_keys is None:
            self._dense_keys = key_states
        else:
            self._dense_keys = torch.cat((self._dense_keys, key_states), dim=-2)
        if self._dense_values is None:
            self._dense_values = value_states
        else:
            self._dense_values = torch.cat((self._dense_values, value_states), dim=-2)
        parts_k = [p for p in (packed_keys, self._dense_keys) if p is not None]
        parts_v = [p for p in (packed_values, self._dense_values) if p is not None]
        return torch.cat(parts_k, dim=-2), torch.cat(parts_v, dim=-2)

    def _update_flat(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        previous_keys = (
            self._decode_group(self.keys_packed, self.rotation, self.centers)
            if self.keys_packed is not None
            else key_states.new_empty((*key_states.shape[:2], 0, key_states.shape[-1]))
        )
        previous_values = (
            self._decode_group(self.values_packed, self.rotation, self.centers)
            if self.values_packed is not None
            else value_states.new_empty((*value_states.shape[:2], 0, value_states.shape[-1]))
        )
        new_keys_packed = self._encode_group(
            key_states, self.rotation, self.centers, self.boundaries, self.bits,
        )
        new_values_packed = self._encode_group(
            value_states, self.rotation, self.centers, self.boundaries, self.bits,
        )
        self.keys_packed = (
            new_keys_packed if self.keys_packed is None
            else self.keys_packed.append(new_keys_packed)
        )
        self.values_packed = (
            new_values_packed if self.values_packed is None
            else self.values_packed.append(new_values_packed)
        )
        return (
            torch.cat((previous_keys, key_states), dim=-2),
            torch.cat((previous_values, value_states), dim=-2),
        )

    def _update_flat_qjl(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        previous_keys = self._decode_keys_full()
        if previous_keys is None:
            previous_keys = key_states.new_empty((*key_states.shape[:2], 0, key_states.shape[-1]))
        previous_values = (
            self._decode_group(self.values_packed, self.rotation, self.centers)
            if self.values_packed is not None
            else value_states.new_empty((*value_states.shape[:2], 0, value_states.shape[-1]))
        )

        new_keys_mse, new_keys_qjl = self._encode_keys_qjl_group(
            key_states,
            self._key_rotation, self._key_centers, self._key_boundaries,
            self.bits - 1, self._qjl_matrix,
        )
        new_values_packed = self._encode_group(
            value_states, self.rotation, self.centers, self.boundaries, self.bits,
        )

        self.keys_packed = (
            new_keys_mse if self.keys_packed is None
            else self.keys_packed.append(new_keys_mse)
        )
        self._keys_qjl = (
            new_keys_qjl if self._keys_qjl is None
            else self._keys_qjl.append(new_keys_qjl)
        )
        self.values_packed = (
            new_values_packed if self.values_packed is None
            else self.values_packed.append(new_values_packed)
        )

        return (
            torch.cat((previous_keys, key_states), dim=-2),
            torch.cat((previous_values, value_states), dim=-2),
        )

    def _update_split_qjl(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        previous_keys = self._decode_keys_full()
        if previous_keys is None:
            previous_keys = key_states.new_empty((*key_states.shape[:2], 0, key_states.shape[-1]))

        if self.values_packed is not None and self._values_outlier is not None:
            previous_values = self._decode_merge(self.values_packed, self._values_outlier)
        else:
            previous_values = value_states.new_empty((*value_states.shape[:2], 0, value_states.shape[-1]))

        new_k_mse, new_k_qjl, new_ko_mse, new_ko_qjl = self._encode_keys_split_qjl(key_states)
        new_normal_values, new_outlier_values = self._encode_split(value_states)

        self.keys_packed = new_k_mse if self.keys_packed is None else self.keys_packed.append(new_k_mse)
        self._keys_qjl = new_k_qjl if self._keys_qjl is None else self._keys_qjl.append(new_k_qjl)
        self._keys_outlier = new_ko_mse if self._keys_outlier is None else self._keys_outlier.append(new_ko_mse)
        self._keys_outlier_qjl = (
            new_ko_qjl if self._keys_outlier_qjl is None
            else self._keys_outlier_qjl.append(new_ko_qjl)
        )
        self.values_packed = (
            new_normal_values if self.values_packed is None
            else self.values_packed.append(new_normal_values)
        )
        self._values_outlier = (
            new_outlier_values if self._values_outlier is None
            else self._values_outlier.append(new_outlier_values)
        )

        return (
            torch.cat((previous_keys, key_states), dim=-2),
            torch.cat((previous_values, value_states), dim=-2),
        )

    def _update_split(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.keys_packed is not None and self._keys_outlier is not None:
            previous_keys = self._decode_merge(self.keys_packed, self._keys_outlier)
        else:
            previous_keys = key_states.new_empty((*key_states.shape[:2], 0, key_states.shape[-1]))

        if self.values_packed is not None and self._values_outlier is not None:
            previous_values = self._decode_merge(self.values_packed, self._values_outlier)
        else:
            previous_values = value_states.new_empty((*value_states.shape[:2], 0, value_states.shape[-1]))

        new_normal_keys, new_outlier_keys = self._encode_split(key_states)
        new_normal_values, new_outlier_values = self._encode_split(value_states)

        self.keys_packed = (
            new_normal_keys if self.keys_packed is None
            else self.keys_packed.append(new_normal_keys)
        )
        self._keys_outlier = (
            new_outlier_keys if self._keys_outlier is None
            else self._keys_outlier.append(new_outlier_keys)
        )
        self.values_packed = (
            new_normal_values if self.values_packed is None
            else self.values_packed.append(new_normal_values)
        )
        self._values_outlier = (
            new_outlier_values if self._values_outlier is None
            else self._values_outlier.append(new_outlier_values)
        )

        return (
            torch.cat((previous_keys, key_states), dim=-2),
            torch.cat((previous_values, value_states), dim=-2),
        )

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        kv_offset = 0
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        packed_len = int(self.keys_packed.original_shape[-2]) if self.keys_packed is not None else 0
        dense_len = int(self._dense_keys.shape[-2]) if self._dense_keys is not None else 0
        return packed_len + dense_len

    def get_max_cache_shape(self) -> int:
        return -1

    def storage_bytes(self) -> int:
        total = 0
        for packed in (self.keys_packed, self.values_packed, self._keys_outlier, self._values_outlier):
            if packed is not None:
                total += packed.storage_bytes()
        for qjl in (self._keys_qjl, self._keys_outlier_qjl):
            if qjl is not None:
                total += qjl.storage_bytes()
        for dense in (self._dense_keys, self._dense_values):
            if dense is not None:
                total += dense.numel() * dense.element_size()
        return int(total)


_APPROX_DMSE_UNIT: dict[int, float] = {
    2: 0.079, 3: 0.030, 4: 0.009, 5: 0.003,
    6: 0.0008, 7: 0.0002, 8: 0.00005,
}


def _auto_norm_threshold(bits: int, head_dim: int, mse_budget: float = 1.0) -> float:
    """Max mean-key-norm a layer can have before we keep it dense.

    Derived from: expected_per_element_mse = norm^2 * D_mse / dim < mse_budget
    """
    d_mse = _APPROX_DMSE_UNIT.get(bits, 0.03)
    return sqrt(mse_budget * head_dim / d_mse)


def build_packed_mse_cache(
    past_key_values,
    bits: int,
    seed: int = 0,
    grid_size: int = 32769,
    num_outlier_channels: int = 0,
    outlier_extra_bits: int = 1,
    use_qjl_keys: bool = False,
    quantize_decode: bool = False,
    norm_guard: bool = True,
) -> Cache:
    import torch as _torch

    layer_data: list[tuple[_torch.Tensor, _torch.Tensor]] = []
    key_norms: list[float] = []
    for key_states, value_states, *_ in past_key_values:
        mean_norm = key_states.float().norm(dim=-1).mean().item()
        key_norms.append(mean_norm)
        layer_data.append((key_states, value_states))

    head_dim = layer_data[0][0].shape[-1] if layer_data else 128
    threshold = _auto_norm_threshold(bits, head_dim) if norm_guard else float("inf")
    num_dense = sum(1 for n in key_norms if n > threshold)

    layers: list[PackedMSELayer] = []
    for idx, (key_states, value_states) in enumerate(layer_data):
        force_dense = key_norms[idx] > threshold
        layer = PackedMSELayer(
            bits=bits,
            seed=seed,
            grid_size=grid_size,
            num_outlier_channels=num_outlier_channels,
            outlier_extra_bits=outlier_extra_bits,
            use_qjl_keys=use_qjl_keys,
            quantize_decode=quantize_decode,
        )
        layer.initialize_from_dense(key_states, value_states, force_dense=force_dense)
        layers.append(layer)

    if num_dense > 0:
        import sys
        dense_ids = [i for i, n in enumerate(key_norms) if n > threshold]
        print(
            f"[TurboQuant] norm_guard: {num_dense}/{len(layers)} layers kept dense "
            f"(threshold={threshold:.1f}, layers={dense_ids})",
            file=sys.stderr,
        )

    return Cache(layers=layers)


def verify_packed_reconstruction(
    original_cache,
    packed_cache: Cache,
) -> list[dict[str, Any]]:
    """Compare reconstructed packed KVs against original dense KVs.

    Returns per-layer metrics so callers can verify reconstruction quality.
    """
    results: list[dict[str, Any]] = []
    for layer_idx, (key_orig, val_orig, *_) in enumerate(original_cache):
        packed_layer: PackedMSELayer = packed_cache.layers[layer_idx]
        key_recon = packed_layer._decode_keys_full()
        val_recon = packed_layer._decode_values_full()

        is_dense = key_recon is None and packed_layer._dense_keys is not None
        if is_dense:
            key_recon = packed_layer._dense_keys
            val_recon = packed_layer._dense_values

        ko = key_orig.float()
        kr = key_recon.float()
        vo = val_orig.float()
        vr = val_recon.float()

        key_norm = ko.norm(dim=-1).mean().item()
        val_norm = vo.norm(dim=-1).mean().item()

        key_err = ko - kr
        val_err = vo - vr
        key_mse = (key_err ** 2).mean().item()
        val_mse = (val_err ** 2).mean().item()

        key_cos = torch.nn.functional.cosine_similarity(
            ko.reshape(-1, ko.shape[-1]),
            kr.reshape(-1, kr.shape[-1]),
            dim=-1,
        ).mean().item()
        val_cos = torch.nn.functional.cosine_similarity(
            vo.reshape(-1, vo.shape[-1]),
            vr.reshape(-1, vr.shape[-1]),
            dim=-1,
        ).mean().item()

        results.append({
            "layer": layer_idx,
            "dense": is_dense,
            "key_mse": round(key_mse, 6),
            "key_cosine_sim": round(key_cos, 6),
            "key_max_abs_error": round(key_err.abs().max().item(), 6),
            "key_mean_norm": round(key_norm, 4),
            "val_mse": round(val_mse, 6),
            "val_cosine_sim": round(val_cos, 6),
            "val_max_abs_error": round(val_err.abs().max().item(), 6),
            "val_mean_norm": round(val_norm, 4),
        })
    return results


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
