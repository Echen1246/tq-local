"""Triton kernels for fused TurboQuant decode and attention.

Architecture:

1. ``_unpack_lookup_kernel`` — bit-unpack + codebook gather.
2. ``_dequant_dot_kernel`` — unpack + codebook + dot (key logits only).
3. ``_tile_attention_kernel`` — tile-parallel fused attention with
   optional Q_prod (QJL) support.  Grid = (n_tiles, Q).  Each program
   processes one tile × one query head.

Key identities (Q_mse):
    <q, decode(k)> = norm_k * <q @ R_k^T, key_centers[k_idx]>
    sum(w * decode(v)) = (sum(w * norm_v * val_centers[v_idx])) @ R_v

Q_prod extension (HAS_QJL=True):
    logit += sqrt(π/2)/D * ||r|| * <q @ S^T, signs[k]>
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

    @triton.jit
    def _tile_attention_kernel(
        key_packed_ptr,
        key_norms_ptr,
        val_packed_ptr,
        val_norms_ptr,
        key_centers_ptr,
        val_centers_ptr,
        q_rot_ptr,
        qjl_signs_ptr,
        qjl_rnorms_ptr,
        q_proj_ptr,
        qjl_scale,
        partial_out_ptr,
        partial_max_ptr,
        partial_sum_ptr,
        N,
        head_scale,
        n_groups,
        n_tiles,
        D: tl.constexpr,
        KEY_BITS: tl.constexpr,
        KEY_PACKED_BYTES: tl.constexpr,
        VAL_BITS: tl.constexpr,
        VAL_PACKED_BYTES: tl.constexpr,
        TILE_N: tl.constexpr,
        HAS_QJL: tl.constexpr,
        QJL_PACKED_BYTES: tl.constexpr,
    ):
        tile_id = tl.program_id(0)
        query_id = tl.program_id(1)
        kv_h = query_id // n_groups

        tile_start = tile_id * TILE_N
        n_range = tile_start + tl.arange(0, TILE_N)
        n_valid = n_range < N

        d_range = tl.arange(0, D)
        kv_off_norm = kv_h * N

        # Key logits (MSE component)
        k_bit_off = d_range * KEY_BITS
        k_byte_idx = k_bit_off // 8
        k_bit_in = k_bit_off % 8
        k_mask = (1 << KEY_BITS) - 1
        k_next = tl.minimum(k_byte_idx + 1, KEY_PACKED_BYTES - 1)
        k_spans = k_byte_idx + 1 < KEY_PACKED_BYTES

        kv_off_k = kv_h * N * KEY_PACKED_BYTES
        q_rot = tl.load(q_rot_ptr + query_id * D + d_range)

        k_off = n_range[:, None] * KEY_PACKED_BYTES + kv_off_k
        kb1 = tl.load(key_packed_ptr + k_off + k_byte_idx[None, :],
                       mask=n_valid[:, None], other=0).to(tl.int32)
        kb2 = tl.load(key_packed_ptr + k_off + k_next[None, :],
                       mask=n_valid[:, None] & k_spans[None, :], other=0).to(tl.int32)
        k_idx = ((kb1 >> k_bit_in[None, :]) | (kb2 << (8 - k_bit_in[None, :]))) & k_mask
        k_c = tl.load(key_centers_ptr + k_idx)
        k_norms = tl.load(key_norms_ptr + kv_off_norm + n_range,
                          mask=n_valid, other=0.0).to(tl.float32)

        logits = tl.sum(k_c * q_rot[None, :], axis=1) * k_norms * head_scale

        # QJL correction (Q_prod)
        if HAS_QJL:
            kv_off_qjl = kv_h * N * QJL_PACKED_BYTES
            qjl_byte_idx = d_range // 8
            qjl_bit_in = d_range % 8

            qjl_off = n_range[:, None] * QJL_PACKED_BYTES + kv_off_qjl
            qjl_bytes = tl.load(qjl_signs_ptr + qjl_off + qjl_byte_idx[None, :],
                                mask=n_valid[:, None], other=0).to(tl.int32)
            signs = 2.0 * ((qjl_bytes >> qjl_bit_in[None, :]) & 1).to(tl.float32) - 1.0

            q_proj = tl.load(q_proj_ptr + query_id * D + d_range)
            r_norms = tl.load(qjl_rnorms_ptr + kv_off_norm + n_range,
                              mask=n_valid, other=0.0).to(tl.float32)

            qjl_dot = tl.sum(signs * q_proj[None, :], axis=1)
            logits = logits + qjl_scale * r_norms * qjl_dot * head_scale

        logits = tl.where(n_valid, logits, float("-inf"))

        # Online softmax
        tile_max = tl.max(logits, axis=0)
        exp_logits = tl.exp(logits - tile_max)
        tile_sum = tl.sum(exp_logits, axis=0)

        # Value accumulation
        v_bit_off = d_range * VAL_BITS
        v_byte_idx = v_bit_off // 8
        v_bit_in = v_bit_off % 8
        v_mask = (1 << VAL_BITS) - 1
        v_next = tl.minimum(v_byte_idx + 1, VAL_PACKED_BYTES - 1)
        v_spans = v_byte_idx + 1 < VAL_PACKED_BYTES

        kv_off_v = kv_h * N * VAL_PACKED_BYTES
        v_off = n_range[:, None] * VAL_PACKED_BYTES + kv_off_v
        vb1 = tl.load(val_packed_ptr + v_off + v_byte_idx[None, :],
                       mask=n_valid[:, None], other=0).to(tl.int32)
        vb2 = tl.load(val_packed_ptr + v_off + v_next[None, :],
                       mask=n_valid[:, None] & v_spans[None, :], other=0).to(tl.int32)
        v_idx = ((vb1 >> v_bit_in[None, :]) | (vb2 << (8 - v_bit_in[None, :]))) & v_mask
        v_c = tl.load(val_centers_ptr + v_idx)
        v_norms = tl.load(val_norms_ptr + kv_off_norm + n_range,
                          mask=n_valid, other=0.0).to(tl.float32)

        weights = exp_logits * v_norms
        tile_out = tl.sum(weights[:, None] * v_c, axis=0)

        # Store partials
        out_offset = query_id * n_tiles * D + tile_id * D
        tl.store(partial_out_ptr + out_offset + d_range, tile_out)
        tl.store(partial_max_ptr + query_id * n_tiles + tile_id, tile_max)
        tl.store(partial_sum_ptr + query_id * n_tiles + tile_id, tile_sum)

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

    Supports both Q_mse (use_qjl_keys=False) and Q_prod
    (use_qjl_keys=True).  Returns ``[B, Q, Sq, D]`` or ``None``
    if inapplicable.
    """
    if not HAS_TRITON:
        return None
    if packed_layer._outlier_enabled:
        return None
    if not query_states.is_cuda:
        return None
    if packed_layer.keys_packed is None:
        return None

    from math import pi as _pi, sqrt as _sqrt

    B, Q, Sq, D = query_states.shape
    if Sq != 1 or B != 1:
        return None

    KV = packed_layer.keys_packed.original_shape[1]
    N = packed_layer.keys_packed.original_shape[2]
    groups = Q // KV
    has_qjl = packed_layer.use_qjl_keys and packed_layer._keys_qjl is not None

    if has_qjl:
        key_rotation = packed_layer._key_rotation
        key_centers = packed_layer._key_centers
        key_bits = packed_layer.bits - 1
    else:
        key_rotation = packed_layer.rotation
        key_centers = packed_layer.centers
        key_bits = packed_layer.bits

    val_rotation = packed_layer.rotation
    val_centers = packed_layer.centers
    val_bits = packed_layer.bits

    key_pb = packed_layer.keys_packed.packed_indices.shape[-1]
    val_pb = packed_layer.values_packed.packed_indices.shape[-1]
    head_scale = 1.0 / _sqrt(D)
    tile_n = 64
    n_tiles = (N + tile_n - 1) // tile_n

    q_rot = query_states.float().squeeze(0).squeeze(1) @ key_rotation.T

    dev = query_states.device
    partial_out = torch.empty(Q, n_tiles, D, device=dev, dtype=torch.float32)
    partial_max = torch.empty(Q, n_tiles, device=dev, dtype=torch.float32)
    partial_sum = torch.empty(Q, n_tiles, device=dev, dtype=torch.float32)

    kp_flat = packed_layer.keys_packed.packed_indices[0].reshape(KV * N, key_pb).contiguous()
    kn_flat = packed_layer.keys_packed.norms[0].reshape(KV * N).contiguous()
    vp_flat = packed_layer.values_packed.packed_indices[0].reshape(KV * N, val_pb).contiguous()
    vn_flat = packed_layer.values_packed.norms[0].reshape(KV * N).contiguous()

    if has_qjl:
        qjl_pb = packed_layer._keys_qjl.packed_signs.shape[-1]
        qjl_signs_flat = packed_layer._keys_qjl.packed_signs[0].reshape(KV * N, qjl_pb).contiguous()
        qjl_rnorms_flat = packed_layer._keys_qjl.residual_norms[0].reshape(KV * N).contiguous()
        q_proj = (query_states.float().squeeze(0).squeeze(1)
                  @ packed_layer._qjl_matrix.T).contiguous()
        qjl_scale = _sqrt(_pi / 2) / D
    else:
        qjl_pb = 1
        qjl_signs_flat = torch.empty(1, 1, dtype=torch.uint8, device=dev)
        qjl_rnorms_flat = torch.empty(1, dtype=torch.float16, device=dev)
        q_proj = torch.empty(Q, D, dtype=torch.float32, device=dev)
        qjl_scale = 0.0

    _tile_attention_kernel[(n_tiles, Q)](
        kp_flat, kn_flat, vp_flat, vn_flat,
        key_centers, val_centers,
        q_rot.contiguous(),
        qjl_signs_flat, qjl_rnorms_flat, q_proj, qjl_scale,
        partial_out, partial_max, partial_sum,
        N, head_scale,
        groups, n_tiles,
        D=D,
        KEY_BITS=key_bits, KEY_PACKED_BYTES=key_pb,
        VAL_BITS=val_bits, VAL_PACKED_BYTES=val_pb,
        TILE_N=tile_n,
        HAS_QJL=has_qjl, QJL_PACKED_BYTES=qjl_pb,
    )

    # Online softmax reduction across tiles
    global_max = partial_max.max(dim=1, keepdim=True).values
    correction = torch.exp(partial_max - global_max)
    corrected_sum = (partial_sum * correction).sum(dim=1)
    corrected_out = (partial_out * correction.unsqueeze(2)).sum(dim=1)

    out_rot = corrected_out / corrected_sum.unsqueeze(1).clamp(min=1e-8)

    # Merge with dense decode buffer if present
    dk = packed_layer._dense_keys
    dv = packed_layer._dense_values
    if dk is not None and dk.shape[-2] > 0:
        dk_f = dk.float()
        dv_f = dv.float()
        q_f = query_states.float()
        dense_logits = (q_f @ dk_f.transpose(-2, -1)).view(Q, -1) * head_scale

        dense_max = dense_logits.max(dim=-1).values
        packed_max_scalar = global_max.squeeze(1)
        combined_max = torch.maximum(packed_max_scalar, dense_max)
        packed_corr = torch.exp(packed_max_scalar - combined_max)

        dense_exp = torch.exp(dense_logits - combined_max.unsqueeze(1))
        dense_sum = dense_exp.sum(dim=-1)

        combined_sum = corrected_sum * packed_corr + dense_sum

        packed_contrib = (out_rot * corrected_sum.unsqueeze(1) * packed_corr.unsqueeze(1)) @ val_rotation
        n_kv = dk.shape[1]
        n_dense = dk.shape[2]
        dv_for_attn = dv_f.view(1, n_kv, 1, n_dense, D).expand(1, n_kv, groups, n_dense, D)
        dv_for_attn = dv_for_attn.reshape(1, Q, n_dense, D).squeeze(0)
        dense_contrib = (dense_exp.unsqueeze(-1) * dv_for_attn).sum(dim=-2)

        output = (packed_contrib + dense_contrib) / combined_sum.unsqueeze(1).clamp(min=1e-8)
    else:
        output = out_rot @ val_rotation

    return output.to(query_states.dtype).view(B, Q, Sq, D)
