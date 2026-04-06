"""TurboQuant attention kernels — both standalone and HF-integrated.

Provides two interfaces:

1. ``qprod_attention`` — standalone function that computes attention from
   the two-part Q_prod compressed key representation.

2. ``turboquant_attention_forward`` — registered as an HF Transformers
   ``AttentionInterface`` backend so the model's own forward pass uses
   compressed KV directly, avoiding full decompression in ``cache.update()``.

The TurboQuant paper prescribes decode-time attention as:

    logit[q, k] = <q, k_hat_mse> + sqrt(pi/2)/d * ||r|| * <Sq, qjl>

The first term is a standard inner product with the MSE-reconstructed key.
The second term is a scalar QJL correction that makes the combined logit
an UNBIASED estimate of <q, k>.
"""

from __future__ import annotations

from math import pi, sqrt
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from turboquant.runtime.packed_qmse_cache import PackedMSELayer


def _repeat_kv(tensor: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return tensor
    return tensor.repeat_interleave(n_rep, dim=1)


# ---------------------------------------------------------------------------
# HF Transformers AttentionInterface integration
# ---------------------------------------------------------------------------

def turboquant_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """HF-compatible attention that works directly from compressed KV cache.

    When ``module._tq_cache_layer`` is set (a ``PackedMSELayer``), this
    computes attention from the compressed representation, avoiding the
    full decompression that the normal ``cache.update()`` path requires.

    When no compressed cache is attached (e.g. during prefill), falls
    through to standard ``F.scaled_dot_product_attention``.
    """
    cache_layer: PackedMSELayer | None = getattr(module, "_tq_cache_layer", None)

    if cache_layer is None or cache_layer._force_dense:
        return _sdpa_fallback(module, query, key, value, attention_mask, scaling)

    n_kv_groups = getattr(module, "num_key_value_groups", 1)

    # Try fully fused Triton attention first (single kernel for K+V+softmax).
    try:
        from turboquant.runtime.triton_kernels import fused_attention
        output = fused_attention(query, cache_layer, n_kv_groups, attention_mask)
    except ImportError:
        output = None
    if output is None:
        output = chunked_turboquant_attention(
            query_states=query,
            packed_layer=cache_layer,
            new_key=None,
            new_value=None,
            n_kv_groups=n_kv_groups,
            attention_mask=attention_mask,
        )
    # chunked_turboquant_attention returns [B, Q_heads, Sq, D]; HF expects [B, Sq, Q_heads, D]
    output = output.transpose(1, 2).contiguous()
    return output, None


def _sdpa_fallback(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None,
) -> tuple[torch.Tensor, None]:
    """Standard SDPA path for prefill or force-dense layers."""
    n_kv_groups = getattr(module, "num_key_value_groups", 1)
    if n_kv_groups > 1:
        key = _repeat_kv(key, n_kv_groups)
        value = _repeat_kv(value, n_kv_groups)

    causal = attention_mask is None and query.shape[-2] > 1
    attn_output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=causal,
        scale=scaling,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


def _register_attention_backend() -> None:
    """Register 'turboquant' with the HF AttentionInterface (idempotent)."""
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        if "turboquant" not in ALL_ATTENTION_FUNCTIONS:
            ALL_ATTENTION_FUNCTIONS.register("turboquant", turboquant_attention_forward)
    except ImportError:
        return

    try:
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        if "turboquant" not in ALL_MASK_ATTENTION_FUNCTIONS:
            sdpa_mask = ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]
            ALL_MASK_ATTENTION_FUNCTIONS.register("turboquant", sdpa_mask)
    except (ImportError, KeyError):
        pass


_register_attention_backend()


# ---------------------------------------------------------------------------
# Chunked online-softmax attention (Phase 2 — real peak VRAM savings)
# ---------------------------------------------------------------------------

_DEFAULT_CHUNK_SIZE = 1024


def _try_triton_key_logits(
    query_states: torch.Tensor,
    packed_layer: "PackedMSELayer",
    start: int,
    end: int,
    n_kv_groups: int,
    head_scale: float,
) -> torch.Tensor | None:
    """Attempt fused Triton dequant+dot for key logits.

    Returns ``[B, Q, Sq, chunk_len]`` logits, or ``None`` if Triton
    is unavailable or the path is unsupported (e.g. QJL / outlier keys).
    """
    if packed_layer.use_qjl_keys or packed_layer._outlier_enabled:
        return None
    if not query_states.is_cuda:
        return None
    try:
        from turboquant.runtime.triton_kernels import triton_available, triton_dequant_dot

        if not triton_available():
            return None
    except Exception:
        return None

    B, Q, Sq, D = query_states.shape
    KV = packed_layer.keys_packed.original_shape[1]
    n_groups = Q // KV
    chunk_len = end - start

    rotation = packed_layer.rotation
    centers = packed_layer.centers
    bits = packed_layer.bits

    q_rot = query_states.float() @ rotation.T

    kp_slice = packed_layer._slice_packed(packed_layer.keys_packed, start, end)
    logits = torch.empty(B, Q, Sq, chunk_len, device=query_states.device, dtype=torch.float32)

    for kv_h in range(KV):
        head_packed = kp_slice.packed_indices[:, kv_h, :, :].reshape(-1, kp_slice.packed_indices.shape[-1])
        head_norms = kp_slice.norms[:, kv_h, :].reshape(-1)

        q_start_h = kv_h * n_groups
        q_end_h = q_start_h + n_groups
        head_q = q_rot[:, q_start_h:q_end_h, :, :].reshape(-1, D).contiguous()

        dots = triton_dequant_dot(
            head_packed.contiguous(),
            head_norms.contiguous(),
            centers,
            head_q,
            bits,
            D,
        )

        logits[:, q_start_h:q_end_h, :, :] = (
            dots.view(B, n_groups, Sq, chunk_len) * head_scale
        )

    return logits


def _online_softmax_update(
    running_max: torch.Tensor,
    running_sum: torch.Tensor,
    running_output: torch.Tensor,
    chunk_logits: torch.Tensor,
    chunk_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One step of online softmax accumulation.

    All tensors are float32. ``chunk_logits`` is [B, Q, Sq, C] and
    ``chunk_values`` is [B, Q, C, D].
    """
    chunk_max = chunk_logits.max(dim=-1, keepdim=True).values  # [B,Q,Sq,1]
    new_max = torch.maximum(running_max, chunk_max)

    correction = torch.exp(running_max - new_max)
    running_sum = running_sum * correction
    running_output = running_output * correction

    exp_logits = torch.exp(chunk_logits - new_max)  # [B,Q,Sq,C]
    running_sum = running_sum + exp_logits.sum(dim=-1, keepdim=True)
    running_output = running_output + exp_logits @ chunk_values.float()

    return new_max, running_sum, running_output


def chunked_turboquant_attention(
    query_states: torch.Tensor,
    packed_layer: "PackedMSELayer",
    new_key: torch.Tensor | None = None,
    new_value: torch.Tensor | None = None,
    n_kv_groups: int = 1,
    attention_mask: torch.Tensor | None = None,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> torch.Tensor:
    """Compute attention from compressed cache using chunked online softmax.

    Unlike ``qprod_attention`` which decompresses the entire history at once,
    this function processes the compressed cache in chunks of ``chunk_size``
    tokens.  Peak decompressed memory per layer is bounded by
    ``chunk_size * D * 2`` (keys + values) regardless of total sequence length.
    """
    B, Q, Sq, D = query_states.shape
    head_scale = 1.0 / sqrt(D)
    q_float = query_states.float()

    NEG_INF = torch.tensor(float("-inf"), device=query_states.device, dtype=torch.float32)
    running_max = NEG_INF.expand(B, Q, Sq, 1).clone()
    running_sum = torch.zeros(B, Q, Sq, 1, device=query_states.device, dtype=torch.float32)
    running_output = torch.zeros(B, Q, Sq, D, device=query_states.device, dtype=torch.float32)

    mask_offset = 0

    # ── Process compressed packed history in chunks ────────────────
    packed_len = packed_layer.packed_seq_length()
    if packed_len > 0:
        for start in range(0, packed_len, chunk_size):
            end = min(start + chunk_size, packed_len)

            triton_logits = _try_triton_key_logits(
                query_states, packed_layer, start, end, n_kv_groups, head_scale,
            )
            if triton_logits is not None:
                logits_chunk = triton_logits
            else:
                keys_chunk = packed_layer._decode_keys_range(start, end)
                keys_chunk = _repeat_kv(keys_chunk, n_kv_groups)
                logits_chunk = (q_float @ keys_chunk.float().transpose(-2, -1)) * head_scale
                del keys_chunk

            vals_chunk = packed_layer._decode_values_range(start, end)
            vals_chunk = _repeat_kv(vals_chunk, n_kv_groups)

            if attention_mask is not None:
                mask_slice = attention_mask[:, :, :, mask_offset:mask_offset + (end - start)]
                logits_chunk = logits_chunk + mask_slice.float()

            running_max, running_sum, running_output = _online_softmax_update(
                running_max, running_sum, running_output, logits_chunk, vals_chunk,
            )
            mask_offset += (end - start)
            del vals_chunk

    # ── Process dense decode buffer ────────────────────────────────
    if packed_layer._dense_keys is not None:
        dk = _repeat_kv(packed_layer._dense_keys, n_kv_groups)
        dv = _repeat_kv(packed_layer._dense_values, n_kv_groups)
        logits_dense = (q_float @ dk.float().transpose(-2, -1)) * head_scale
        dense_len = dk.shape[-2]

        if attention_mask is not None:
            mask_slice = attention_mask[:, :, :, mask_offset:mask_offset + dense_len]
            logits_dense = logits_dense + mask_slice.float()

        running_max, running_sum, running_output = _online_softmax_update(
            running_max, running_sum, running_output, logits_dense, dv,
        )
        mask_offset += dense_len

    # ── Process new token (if any) ─────────────────────────────────
    if new_key is not None:
        nk = _repeat_kv(new_key, n_kv_groups)
        nv = _repeat_kv(new_value, n_kv_groups)
        logits_new = (q_float @ nk.float().transpose(-2, -1)) * head_scale
        new_len = nk.shape[-2]

        if attention_mask is not None:
            mask_slice = attention_mask[:, :, :, mask_offset:mask_offset + new_len]
            logits_new = logits_new + mask_slice.float()

        running_max, running_sum, running_output = _online_softmax_update(
            running_max, running_sum, running_output, logits_new, nv,
        )

    # ── Normalize ──────────────────────────────────────────────────
    output = running_output / running_sum.clamp(min=1e-8)
    return output.to(dtype=query_states.dtype)


# ---------------------------------------------------------------------------
# Standalone Q_prod attention (original interface, still usable directly)
# ---------------------------------------------------------------------------

def qprod_attention(
    query_states: torch.Tensor,
    packed_layer: "PackedMSELayer",
    new_key: torch.Tensor | None = None,
    new_value: torch.Tensor | None = None,
    n_kv_groups: int = 1,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute attention from the Q_prod compressed representation.

    Instead of materializing ``k_hat_prod`` and calling SDPA, this builds
    the attention logits in two parts:

    1. **MSE logits** — ``q @ k_hat_mse.T`` from the ``(b-1)``-bit MSE
       quantized keys (decoded on the fly from packed indices).
    2. **QJL logits** — ``sqrt(pi/2)/d * ||r|| * (q @ S^T) @ signs^T``
       from the 1-bit QJL sign vectors and stored residual norms.

    The two are summed, softmaxed, and multiplied by Q_mse-decoded values.

    Parameters
    ----------
    query_states : Tensor [B, Q_heads, Sq, D]
        Queries after projection and RoPE.
    packed_layer : PackedMSELayer
        Cache layer containing compressed keys (Q_prod) and values (Q_mse).
    new_key, new_value : Tensor [B, KV_heads, Sn, D] or None
        Current decode token(s) in full precision.
    n_kv_groups : int
        ``num_query_heads // num_kv_heads`` for Grouped Query Attention.
    attention_mask : Tensor [B, 1, Sq, Stotal] or None
        Additive attention mask (``0`` = attend, ``-inf`` = mask).

    Returns
    -------
    Tensor [B, Q_heads, Sq, D]
    """
    from turboquant.runtime.packed_qmse_cache import (
        PackedMSELayer,
        _unpack_qjl_signs,
    )

    B, Q, Sq, D = query_states.shape
    head_scale = 1.0 / sqrt(D)
    parts_logits: list[torch.Tensor] = []
    parts_values: list[torch.Tensor] = []

    # ── Compressed history (Q_prod keys, Q_mse values) ──────────────
    if packed_layer.keys_packed is not None and packed_layer._keys_qjl is not None:
        kp = packed_layer.keys_packed
        Sc = kp.original_shape[-2]
        KV = kp.original_shape[1]

        # 1a) MSE logits: q @ k_hat_mse^T
        key_mse = PackedMSELayer._decode_group(
            kp, packed_layer._key_rotation, packed_layer._key_centers,
        )
        key_mse = _repeat_kv(key_mse, n_kv_groups)
        logits_mse = query_states.float() @ key_mse.float().transpose(-2, -1)

        # 1b) QJL logits: sqrt(pi/2)/d * ||r|| * <Sq, qjl>
        S = packed_layer._qjl_matrix  # [D, D]
        q_proj = query_states.float() @ S.float().T  # [B, Q, Sq, D]

        num_vec = kp.num_vectors
        signs = _unpack_qjl_signs(
            packed_layer._keys_qjl, num_vec,
        )  # [N, D]  float32
        signs = signs.view(B, KV, Sc, D)
        signs = _repeat_kv(signs, n_kv_groups)

        logits_qjl = q_proj @ signs.transpose(-2, -1)  # [B, Q, Sq, Sc]

        r_norms = packed_layer._keys_qjl.residual_norms.float()  # [B, KV, Sc]
        r_norms = _repeat_kv(r_norms.unsqueeze(2), n_kv_groups).squeeze(2)
        qjl_scale = sqrt(pi / 2) / D
        logits_qjl = logits_qjl * (r_norms.unsqueeze(2) * qjl_scale)

        parts_logits.append((logits_mse + logits_qjl) * head_scale)

        val = packed_layer._decode_values_full()
        parts_values.append(_repeat_kv(val, n_kv_groups))

    elif packed_layer.keys_packed is not None:
        # Q_mse-only keys (no QJL) — standard materialized path
        key_decoded = packed_layer._decode_keys_full()
        key_decoded = _repeat_kv(key_decoded, n_kv_groups)
        parts_logits.append(
            (query_states.float() @ key_decoded.float().transpose(-2, -1)) * head_scale,
        )
        val = packed_layer._decode_values_full()
        parts_values.append(_repeat_kv(val, n_kv_groups))

    # ── Dense buffer (force_dense layers or dense-decode tokens) ────
    if packed_layer._dense_keys is not None:
        dk = _repeat_kv(packed_layer._dense_keys, n_kv_groups)
        parts_logits.append(
            (query_states.float() @ dk.float().transpose(-2, -1)) * head_scale,
        )
        dv = _repeat_kv(packed_layer._dense_values, n_kv_groups)
        parts_values.append(dv)

    # ── New token (current decode step, full precision) ─────────────
    if new_key is not None:
        nk = _repeat_kv(new_key, n_kv_groups)
        parts_logits.append(
            (query_states.float() @ nk.float().transpose(-2, -1)) * head_scale,
        )
        parts_values.append(_repeat_kv(new_value, n_kv_groups))

    # ── Combine, softmax, weighted sum ──────────────────────────────
    all_logits = torch.cat(parts_logits, dim=-1)
    all_values = torch.cat(parts_values, dim=-2)

    if attention_mask is not None:
        all_logits = all_logits + attention_mask.float()

    attn_weights = torch.softmax(all_logits, dim=-1, dtype=torch.float32)
    attn_weights = attn_weights.to(dtype=query_states.dtype)
    return attn_weights @ all_values
