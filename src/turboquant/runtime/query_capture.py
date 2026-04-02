from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from safetensors.torch import save_file


def _layer_list(model):
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise AttributeError("Expected model.model.layers for Qwen/QwQ attention capture.")
    return model.model.layers


def capture_query_projections(model, forward_fn: Callable[[], object]) -> dict[str, object]:
    layers = _layer_list(model)
    num_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // num_heads
    captured: dict[str, object] = {}
    handles = []

    for layer_index, layer in enumerate(layers):
        def hook(_module, _inputs, output, idx=layer_index):
            tensor = output[0] if isinstance(output, tuple) else output
            batch, seq_len, _ = tensor.shape
            reshaped = tensor.detach().to("cpu").reshape(batch, seq_len, num_heads, head_dim)
            captured[f"layer_{idx:02d}.query"] = reshaped.permute(0, 2, 1, 3).contiguous()

        handles.append(layer.self_attn.q_proj.register_forward_hook(hook))

    try:
        forward_output = forward_fn()
    finally:
        for handle in handles:
            handle.remove()

    return captured, forward_output


def save_query_projections(path: Path, state_dict: dict[str, object]) -> None:
    save_file(state_dict, str(path))
