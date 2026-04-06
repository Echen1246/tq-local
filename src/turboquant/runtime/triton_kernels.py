"""Triton kernels for fused TurboQuant decode and attention.

Architecture:

1. ``_unpack_lookup_kernel`` — bit-unpack + codebook gather.
2. ``_dequant_dot_kernel`` — unpack + codebook + dot (key logits only).
3. ``_tile_attention_kernel`` — tile-parallel fused attention.
   Grid = (n_tiles, Q). Each program = one tile x one query head.
   Partial results reduced in Python (cheaper than a Triton reduction
   kernel for the small [Q, n_tiles] tensors involved).

Key identities:
    <q, decode(k)> = norm_k * <q @ R^T, centers[k_idx]>
    sum(w * decode(v)) = (sum(w * norm_v * centers[v_idx])) @ R
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def triton_available() -> bool:
    return HAS_TRITON


if HAS_TRITON:

    @triton.jit
    def _unpack_lookup_kernel(
        packed_ptr, centers_ptr, output_ptr,
        N,
        D: tl.constexpr, BITS: tl.constexpr, PACKED_BYTES: tl.constexpr,
    ):
        n_id = tl.program_id(0)
        if n_id >= N:
            return
        d_range = tl.arange(0, D)
        bit_offsets = d_range * BITS
        byte_idxs = bit_offsets // 8
        bit_in_bytes = bit_offsets % 8
        base = n_id * PACKED_BYTES
        b1 = tl.load(packed_ptr + base + byte_idxs).to(tl.int32)
        nx = tl.minimum(byte_idxs + 1, PACKED_BYTES - 1)
        b2 = tl.load(packed_ptr + base + nx).to(tl.int32)
        b2 = tl.where(byte_idxs + 1 < PACKED_BYTES, b2, 0)
        idx = ((b1 >> bit_in_bytes) | (b2 << (8 - bit_in_bytes))) & ((1 << BITS) - 1)
        tl.store(output_ptr + n_id * D + d_range, tl.load(centers_ptr + idx))

    @triton.jit
    def _dequant_dot_kernel(
        packed_ptr, norms_ptr, centers_ptr, q_rot_ptr, output_ptr,
        N, Q,
        D: tl.constexpr, BITS: tl.constexpr, PACKED_BYTES: tl.constexpr,
    ):
        n_id = tl.program_id(0)
        if n_id >= N:
            return
        d_range = tl.arange(0, D)
        bit_offsets = d_range * BITS
        byte_idxs = bit_offsets // 8
        bit_in_bytes = bit_offsets % 8
        base = n_id * PACKED_BYTES
        b1 = tl.load(packed_ptr + base + byte_idxs).to(tl.int32)
        nx = tl.minimum(byte_idxs + 1, PACKED_BYTES - 1)
        b2 = tl.load(packed_ptr + base + nx).to(tl.int32)
        b2 = tl.where(byte_idxs + 1 < PACKED_BYTES, b2, 0)
        idx = ((b1 >> bit_in_bytes) | (b2 << (8 - bit_in_bytes))) & ((1 << BITS) - 1)
        c = tl.load(centers_ptr + idx)
        norm = tl.load(norms_ptr + n_id).to(tl.float32)
        q_id = tl.zeros((), dtype=tl.int32)
        while q_id < Q:
            q_vals = tl.load(q_rot_ptr + q_id * D + d_range)
            tl.store(output_ptr + q_id * N + n_id, tl.sum(c * q_vals) * norm)
            q_id += 1

    # ---------------------------------------------------------------
    # Tile-parallel fused attention kernel
    #
    # Grid: (n_tiles, Q)
    # Each program processes exactly TILE_N positions for one query
    # head.  No loops — straight-line GPU code.
    #
    # Tuning notes (Llama 3.1 8B, D=128, 4-bit, B200):
    #   TILE_N=64, num_warps=4  →  best (6.70s / 128 tok = 2.2x)
    #   TILE_N=128              →  worse (register pressure)
    #   TILE_N=32               →  worse (too many programs)
    #   num_warps=8             →  worse (warp scheduling overhead)
    #   GQA-fused kernel        →  worse (low occupancy from
    #                              keeping K+V in registers across
    #                              the group loop)
    #   Triton reduce kernel    →  neutral (JIT cost offsets the
    #                              savings from fewer PyTorch ops)
    # ---------------------------------------------------------------

    @triton.jit
    def _tile_attention_kernel(
        key_packed_ptr,
        key_norms_ptr,
        val_packed_ptr,
        val_norms_ptr,
        centers_ptr,
        q_rot_ptr,
        partial_out_ptr,
        partial_max_ptr,
        partial_sum_ptr,
        N,
        head_scale,
        kv_stride_packed,
        kv_stride_norm,
        n_groups,
        n_tiles,
        D: tl.constexpr,
        BITS: tl.constexpr,
        PACKED_BYTES: tl.constexpr,
        TILE_N: tl.constexpr,
    ):
        tile_id = tl.program_id(0)
        query_id = tl.program_id(1)
        kv_h = query_id // n_groups

        tile_start = tile_id * TILE_N
        n_range = tile_start + tl.arange(0, TILE_N)
        n_valid = n_range < N

        d_range = tl.arange(0, D)
        bit_offsets = d_range * BITS
        byte_idxs = bit_offsets // 8
        bit_in_bytes = bit_offsets % 8
        bit_mask = (1 << BITS) - 1
        next_byte = tl.minimum(byte_idxs + 1, PACKED_BYTES - 1)
        spans_byte = byte_idxs + 1 < PACKED_BYTES

        kv_off_packed = kv_h * kv_stride_packed
        kv_off_norm = kv_h * kv_stride_norm

        q_rot = tl.load(q_rot_ptr + query_id * D + d_range)

        k_off = n_range[:, None] * PACKED_BYTES + kv_off_packed
        kb1 = tl.load(key_packed_ptr + k_off + byte_idxs[None, :],
                       mask=n_valid[:, None], other=0).to(tl.int32)
        kb2 = tl.load(key_packed_ptr + k_off + next_byte[None, :],
                       mask=n_valid[:, None] & spans_byte[None, :], other=0).to(tl.int32)
        k_idx = ((kb1 >> bit_in_bytes[None, :]) | (kb2 << (8 - bit_in_bytes[None, :]))) & bit_mask
        k_c = tl.load(centers_ptr + k_idx)
        k_norms = tl.load(key_norms_ptr + kv_off_norm + n_range,
                          mask=n_valid, other=0.0).to(tl.float32)

        logits = tl.sum(k_c * q_rot[None, :], axis=1) * k_norms * head_scale
        logits = tl.where(n_valid, logits, float("-inf"))

        tile_max = tl.max(logits, axis=0)
        exp_logits = tl.exp(logits - tile_max)
        tile_sum = tl.sum(exp_logits, axis=0)

        v_off = n_range[:, None] * PACKED_BYTES + kv_off_packed
        vb1 = tl.load(val_packed_ptr + v_off + byte_idxs[None, :],
                       mask=n_valid[:, None], other=0).to(tl.int32)
        vb2 = tl.load(val_packed_ptr + v_off + next_byte[None, :],
                       mask=n_valid[:, None] & spans_byte[None, :], other=0).to(tl.int32)
        v_idx = ((vb1 >> bit_in_bytes[None, :]) | (vb2 << (8 - bit_in_bytes[None, :]))) & bit_mask
        v_c = tl.load(centers_ptr + v_idx)
        v_norms = tl.load(val_norms_ptr + kv_off_norm + n_range,
                          mask=n_valid, other=0.0).to(tl.float32)

        weights = exp_logits * v_norms
        tile_out = tl.sum(weights[:, None] * v_c, axis=0)

        out_offset = query_id * n_tiles * D + tile_id * D
        tl.store(partial_out_ptr + out_offset + d_range, tile_out)
        tl.store(partial_max_ptr + query_id * n_tiles + tile_id, tile_max)
        tl.store(partial_sum_ptr + query_id * n_tiles + tile_id, tile_sum)


# -------------------------------------------------------------------
# Python wrappers
# -------------------------------------------------------------------


def triton_unpack_lookup(
    packed_indices: torch.Tensor, centers: torch.Tensor,
    bits: int, dim: int,
) -> torch.Tensor:
    if not HAS_TRITON:
        raise RuntimeError("Triton is not installed.")
    N, pb = packed_indices.shape
    out = torch.empty(N, dim, dtype=torch.float32, device=packed_indices.device)
    _unpack_lookup_kernel[(N,)](packed_indices, centers, out, N, D=dim, BITS=bits, PACKED_BYTES=pb)
    return out


def triton_dequant_dot(
    packed_indices: torch.Tensor, norms: torch.Tensor,
    centers: torch.Tensor, q_rot: torch.Tensor,
    bits: int, dim: int,
) -> torch.Tensor:
    if not HAS_TRITON:
        raise RuntimeError("Triton is not installed.")
    N, pb = packed_indices.shape
    Q = q_rot.shape[0]
    out = torch.empty(Q, N, dtype=torch.float32, device=packed_indices.device)
    _dequant_dot_kernel[(N,)](
        packed_indices, norms.contiguous(), centers, q_rot.contiguous(), out,
        N, Q, D=dim, BITS=bits, PACKED_BYTES=pb,
    )
    return out


def triton_decode_group(
    packed_indices: torch.Tensor, norms: torch.Tensor,
    rotation: torch.Tensor, centers: torch.Tensor,
    bits: int, dim: int,
    original_shape: tuple[int, ...], original_dtype: torch.dtype,
) -> torch.Tensor:
    N = packed_indices.shape[0]
    coords = triton_unpack_lookup(packed_indices.view(N, -1).contiguous(), centers, bits, dim)
    norms_flat = norms.view(-1).to(torch.float32)
    return ((coords @ rotation) * norms_flat.unsqueeze(1)).view(original_shape).to(original_dtype)


def fused_attention(
    query_states: torch.Tensor,
    packed_layer,
    n_kv_groups: int,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Full fused TurboQuant attention via tile-parallel Triton kernel.

    Returns ``[B, Q, Sq, D]`` or ``None`` if inapplicable.
    """
    if not HAS_TRITON:
        return None
    if packed_layer.use_qjl_keys or packed_layer._outlier_enabled:
        return None
    if not query_states.is_cuda:
        return None
    if packed_layer.keys_packed is None:
        return None

    from math import sqrt as _sqrt

    B, Q, Sq, D = query_states.shape
    if Sq != 1 or B != 1:
        return None

    KV = packed_layer.keys_packed.original_shape[1]
    N = packed_layer.keys_packed.original_shape[2]
    groups = Q // KV
    rotation = packed_layer.rotation
    centers = packed_layer.centers
    bits = packed_layer.bits
    pb = packed_layer.keys_packed.packed_indices.shape[-1]
    head_scale = 1.0 / _sqrt(D)
    tile_n = 64
    n_tiles = (N + tile_n - 1) // tile_n

    q_rot = query_states.float().squeeze(0).squeeze(1) @ rotation.T  # [Q, D]

    dev = query_states.device
    partial_out = torch.empty(Q, n_tiles, D, device=dev, dtype=torch.float32)
    partial_max = torch.empty(Q, n_tiles, device=dev, dtype=torch.float32)
    partial_sum = torch.empty(Q, n_tiles, device=dev, dtype=torch.float32)

    kp_flat = packed_layer.keys_packed.packed_indices[0].reshape(KV * N, pb).contiguous()
    kn_flat = packed_layer.keys_packed.norms[0].reshape(KV * N).contiguous()
    vp_flat = packed_layer.values_packed.packed_indices[0].reshape(KV * N, pb).contiguous()
    vn_flat = packed_layer.values_packed.norms[0].reshape(KV * N).contiguous()

    _tile_attention_kernel[(n_tiles, Q)](
        kp_flat, kn_flat, vp_flat, vn_flat,
        centers, q_rot.contiguous(),
        partial_out, partial_max, partial_sum,
        N, head_scale,
        N * pb, N, groups, n_tiles,
        D=D, BITS=bits, PACKED_BYTES=pb, TILE_N=tile_n,
    )

    # Online softmax reduction across tiles
    global_max = partial_max.max(dim=1, keepdim=True).values      # [Q, 1]
    correction = torch.exp(partial_max - global_max)               # [Q, n_tiles]
    corrected_sum = (partial_sum * correction).sum(dim=1)          # [Q]
    corrected_out = (partial_out * correction.unsqueeze(2)).sum(dim=1)  # [Q, D]

    out_rot = corrected_out / corrected_sum.unsqueeze(1).clamp(min=1e-8)

    # Merge with dense decode buffer if present
    dk = packed_layer._dense_keys
    dv = packed_layer._dense_values
    if dk is not None and dk.shape[-2] > 0:
        from turboquant.runtime.attention import _repeat_kv

        dk_exp = _repeat_kv(dk, n_kv_groups).float()
        dv_exp = _repeat_kv(dv, n_kv_groups).float()
        q_f = query_states.float()
        dense_logits = (q_f @ dk_exp.transpose(-2, -1)).view(Q, -1) * head_scale

        dense_max = dense_logits.max(dim=-1).values
        packed_max_scalar = global_max.squeeze(1)
        combined_max = torch.maximum(packed_max_scalar, dense_max)
        packed_corr = torch.exp(packed_max_scalar - combined_max)

        dense_exp = torch.exp(dense_logits - combined_max.unsqueeze(1))
        dense_sum = dense_exp.sum(dim=-1)

        combined_sum = corrected_sum * packed_corr + dense_sum

        packed_contrib = (out_rot * corrected_sum.unsqueeze(1) * packed_corr.unsqueeze(1)) @ rotation
        dv_flat = dv_exp.squeeze(0)
        dense_contrib = (dense_exp.unsqueeze(1) @ dv_flat).squeeze(1)

        output = (packed_contrib + dense_contrib) / combined_sum.unsqueeze(1).clamp(min=1e-8)
    else:
        output = out_rot @ rotation

    return output.to(query_states.dtype).view(B, Q, Sq, D)
