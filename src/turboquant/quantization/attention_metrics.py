from __future__ import annotations

from math import sqrt

import numpy as np


def causal_attention_logit_mse(
    queries: np.ndarray,
    keys: np.ndarray,
    reconstructed_keys: np.ndarray,
) -> float:
    if queries.ndim != 4 or keys.ndim != 4 or reconstructed_keys.ndim != 4:
        raise ValueError("Expected query/key tensors with shape [batch, heads, seq, head_dim].")

    batch, query_heads, seq_len, head_dim = queries.shape
    _, key_heads, key_seq_len, _ = keys.shape
    if key_seq_len != seq_len:
        raise ValueError("Expected query and key tensors to share the same sequence length.")
    if query_heads % key_heads != 0:
        raise ValueError("Expected grouped-query attention with query_heads divisible by key_heads.")

    repeat_factor = query_heads // key_heads
    expanded_keys = np.repeat(keys, repeat_factor, axis=1)
    expanded_reconstructed_keys = np.repeat(reconstructed_keys, repeat_factor, axis=1)

    scale = 1.0 / sqrt(head_dim)
    original_logits = np.einsum("bhid,bhjd->bhij", queries, expanded_keys) * scale
    reconstructed_logits = np.einsum("bhid,bhjd->bhij", queries, expanded_reconstructed_keys) * scale
    diff = original_logits - reconstructed_logits

    causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    masked = diff[:, :, causal_mask]
    return float(np.mean(np.square(masked)))
