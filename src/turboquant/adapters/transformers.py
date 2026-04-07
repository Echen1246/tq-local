from __future__ import annotations

from turboquant.constants import DEFAULT_MAX_NEW_TOKENS

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache

from turboquant.runtime.generation import GenerationOutput, greedy_decode_with_prefill_cache
from turboquant.telemetry import summarize_generation_metrics


@dataclass(frozen=True)
class TransformersLoadConfig:
    model_id_or_path: str
    revision: str | None = None
    dtype: str | torch.dtype = "auto"
    device_map: str | dict[str, Any] = "auto"
    attn_implementation: str = "sdpa"
    trust_remote_code: bool = False
    token: str | None = None
    cache_dir: str | None = None


@dataclass(frozen=True)
class CompatibilityReport:
    compatible: bool
    backend: str
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_transformers_model(config: TransformersLoadConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id_or_path,
        revision=config.revision,
        token=config.token,
        cache_dir=config.cache_dir,
        trust_remote_code=config.trust_remote_code,
    )
    torch_dtype = config.dtype
    if isinstance(torch_dtype, str) and torch_dtype != "auto":
        torch_dtype = getattr(torch, torch_dtype, torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id_or_path,
        revision=config.revision,
        token=config.token,
        cache_dir=config.cache_dir,
        torch_dtype=torch_dtype,
        device_map=config.device_map,
        attn_implementation=config.attn_implementation,
        low_cpu_mem_usage=True,
        trust_remote_code=config.trust_remote_code,
    )
    model.eval()
    return tokenizer, model


def inspect_transformers_model_compatibility(model) -> CompatibilityReport:
    config = model.config
    reasons: list[str] = []
    warnings: list[str] = []
    details = {
        "model_type": getattr(config, "model_type", None),
        "is_encoder_decoder": bool(getattr(config, "is_encoder_decoder", False)),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "sliding_window": getattr(config, "sliding_window", None),
        "attention_chunk_size": getattr(config, "attention_chunk_size", None),
        "has_generate": hasattr(model, "generate"),
        "has_model_layers": bool(hasattr(model, "model") and hasattr(model.model, "layers")),
        "use_cache": getattr(config, "use_cache", True),
    }

    if not details["has_generate"]:
        reasons.append("Model does not expose a generate() method.")
    if details["is_encoder_decoder"]:
        reasons.append("Encoder-decoder models are not supported in v1.")
    if not details["use_cache"]:
        reasons.append("Model config disables use_cache.")
    if details["num_hidden_layers"] is None or details["num_attention_heads"] is None:
        reasons.append("Model is missing standard decoder attention metadata.")
    if details["sliding_window"] is not None:
        warnings.append("Sliding-window attention is not explicitly supported in v1.")
    if details["attention_chunk_size"] is not None:
        warnings.append("Chunked attention is not explicitly supported in v1.")
    if getattr(config, "num_kv_shared_layers", None) is not None:
        warnings.append("Shared-KV-layer models are not explicitly supported in v1.")

    return CompatibilityReport(
        compatible=not reasons,
        backend="transformers",
        reasons=reasons,
        warnings=warnings,
        details=details,
    )


def enable_turboquant_attention(model, packed_cache: Cache) -> str | None:
    """Wire TurboQuant compressed cache into the model's attention path.

    For each layer, stashes the corresponding ``PackedMSELayer`` on
    ``self_attn._tq_cache_layer`` and enables lazy updates so that
    ``cache.update()`` no longer decompresses the full history.

    Returns the previous ``_attn_implementation`` so it can be restored.
    """
    import turboquant.runtime.attention  # noqa: F401 — triggers registration

    layers = _get_model_layers(model)
    if layers is None:
        raise RuntimeError("Cannot locate model decoder layers for TurboQuant attention wiring.")

    old_impl = getattr(model.config, "_attn_implementation", None)
    model.config._attn_implementation = "turboquant"

    for idx, decoder_layer in enumerate(layers):
        attn_module = decoder_layer.self_attn
        cache_layer = packed_cache.layers[idx]
        attn_module._tq_cache_layer = cache_layer
        cache_layer._lazy_update = True

    return old_impl


def disable_turboquant_attention(model, packed_cache: Cache, old_impl: str | None = "sdpa") -> None:
    """Reverse the wiring done by ``enable_turboquant_attention``."""
    layers = _get_model_layers(model)
    if layers is None:
        return

    model.config._attn_implementation = old_impl or "sdpa"

    for idx, decoder_layer in enumerate(layers):
        attn_module = decoder_layer.self_attn
        if hasattr(attn_module, "_tq_cache_layer"):
            del attn_module._tq_cache_layer
        if idx < len(packed_cache.layers):
            packed_cache.layers[idx]._lazy_update = False


def _get_model_layers(model):
    """Return the list of decoder layers, handling common model wrappers."""
    inner = getattr(model, "model", None)
    if inner is None:
        return None
    return getattr(inner, "layers", None)


def _render_inputs(
    *,
    tokenizer,
    prompt: str | None,
    messages: list[dict[str, str]] | None,
    add_generation_prompt: bool,
):
    if messages is not None:
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("Tokenizer does not support apply_chat_template, so messages input is unavailable.")
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        return tokenizer(rendered, return_tensors="pt")

    if prompt is None:
        raise ValueError("Either prompt or messages must be provided.")
    return tokenizer(prompt, return_tensors="pt")


class TurboQuantSession:
    def __init__(
        self,
        *,
        model,
        tokenizer,
        variant: str = "qmse_packed",
        bits: int = 3,
        rotation_seed: int = 0,
        num_outlier_channels: int = 0,
        outlier_extra_bits: int = 1,
        use_qjl_keys: bool = False,
        quantize_decode: bool = False,
        norm_guard: bool = True,
    ) -> None:
        report = inspect_transformers_model_compatibility(model)
        if not report.compatible:
            raise ValueError(f"Incompatible model for TurboQuantSession: {report.reasons}")
        self.model = model
        self.tokenizer = tokenizer
        self.variant = variant
        self.bits = bits
        self.rotation_seed = rotation_seed
        self.num_outlier_channels = num_outlier_channels
        self.outlier_extra_bits = outlier_extra_bits
        self.use_qjl_keys = use_qjl_keys
        self.quantize_decode = quantize_decode
        self.norm_guard = norm_guard
        self._last_output: GenerationOutput | None = None
        self.compatibility = report

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str,
        *,
        revision: str | None = None,
        dtype: str | torch.dtype = "auto",
        device_map: str | dict[str, Any] = "auto",
        attn_implementation: str = "sdpa",
        trust_remote_code: bool = False,
        token: str | None = None,
        cache_dir: str | None = None,
        variant: str = "qmse_packed",
        bits: int = 3,
        rotation_seed: int = 0,
        num_outlier_channels: int = 0,
        outlier_extra_bits: int = 1,
        use_qjl_keys: bool = False,
        quantize_decode: bool = False,
        norm_guard: bool = True,
    ) -> "TurboQuantSession":
        tokenizer, model = load_transformers_model(
            TransformersLoadConfig(
                model_id_or_path=model_id_or_path,
                revision=revision,
                dtype=dtype,
                device_map=device_map,
                attn_implementation=attn_implementation,
                trust_remote_code=trust_remote_code,
                token=token,
                cache_dir=cache_dir,
            )
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            variant=variant,
            bits=bits,
            rotation_seed=rotation_seed,
            num_outlier_channels=num_outlier_channels,
            outlier_extra_bits=outlier_extra_bits,
            use_qjl_keys=use_qjl_keys,
            quantize_decode=quantize_decode,
            norm_guard=norm_guard,
        )

    def generate(
        self,
        *,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        add_generation_prompt: bool = True,
        return_output: bool = False,
    ) -> str | GenerationOutput:
        inputs = _render_inputs(
            tokenizer=self.tokenizer,
            prompt=prompt,
            messages=messages,
            add_generation_prompt=add_generation_prompt,
        )
        if torch.cuda.is_available():
            inputs = {name: tensor.to("cuda") for name, tensor in inputs.items()}
        output = greedy_decode_with_prefill_cache(
            model=self.model,
            tokenizer=self.tokenizer,
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            variant=self.variant,
            qmse_bits=self.bits,
            rotation_seed=self.rotation_seed,
            num_outlier_channels=self.num_outlier_channels,
            outlier_extra_bits=self.outlier_extra_bits,
            use_qjl_keys=self.use_qjl_keys,
            quantize_decode=self.quantize_decode,
            norm_guard=self.norm_guard,
        )
        self._last_output = output
        if return_output:
            return output
        return output.text

    def last_metrics(self) -> dict[str, Any] | None:
        if self._last_output is None:
            return None
        return self._last_output.metrics.to_dict()

    def last_telemetry(self) -> dict[str, Any] | None:
        if self._last_output is None:
            return None
        return summarize_generation_metrics(self._last_output.metrics).to_dict()

    def print_telemetry(self) -> None:
        if self._last_output is None:
            print("[TurboQuant] No generation data yet.")
            return
        telem = summarize_generation_metrics(self._last_output.metrics)
        print(telem.format(compact=False))

    def compatibility_report(self) -> dict[str, Any]:
        return self.compatibility.to_dict()


@dataclass
class _TurboQuantState:
    """Internal state stashed on the model when TurboQuant is active."""

    bits: int
    rotation_seed: int
    num_outlier_channels: int
    outlier_extra_bits: int
    use_qjl_keys: bool
    quantize_decode: bool
    norm_guard: bool
    original_generate: Any = None
    call_count: int = 0


def activate(
    model,
    tokenizer=None,
    *,
    bits: int = 4,
    rotation_seed: int = 0,
    num_outlier_channels: int = 0,
    outlier_extra_bits: int = 1,
    use_qjl_keys: bool = False,
    quantize_decode: bool = False,
    norm_guard: bool = True,
    quiet: bool = False,
) -> None:
    """Activate TurboQuant on an existing HuggingFace model.

    After calling this, ``model.generate()`` transparently compresses the
    KV cache. No other code changes are needed.

    Args:
        model: A HuggingFace ``AutoModelForCausalLM`` instance.
        tokenizer: Optional tokenizer. If not provided, one will be loaded
            from the model's ``name_or_path``.
        bits: Quantization bit width (2, 3, or 4).
        quiet: If True, suppress the activation banner.
    """
    if hasattr(model, "_tq_state"):
        raise RuntimeError(
            "TurboQuant is already active on this model. "
            "Call turboquant.deactivate(model) first."
        )

    report = inspect_transformers_model_compatibility(model)
    if not report.compatible:
        raise ValueError(f"Model is not compatible with TurboQuant: {report.reasons}")

    if tokenizer is None:
        name_or_path = getattr(model.config, "_name_or_path", None)
        if name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        else:
            raise ValueError(
                "Cannot auto-detect tokenizer. Pass tokenizer= explicitly."
            )

    state = _TurboQuantState(
        bits=bits,
        rotation_seed=rotation_seed,
        num_outlier_channels=num_outlier_channels,
        outlier_extra_bits=outlier_extra_bits,
        use_qjl_keys=use_qjl_keys,
        quantize_decode=quantize_decode,
        norm_guard=norm_guard,
        original_generate=model.generate,
    )
    model._tq_state = state
    model._tq_tokenizer = tokenizer

    import functools

    @functools.wraps(state.original_generate)
    def _tq_generate(input_ids=None, **kwargs):
        tq: _TurboQuantState = model._tq_state
        tok = model._tq_tokenizer

        max_new_tokens = kwargs.pop("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
        attention_mask = kwargs.pop("attention_mask", None)

        if input_ids is None:
            raise ValueError("input_ids is required for TurboQuant generate().")

        inputs = {"input_ids": input_ids}
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask

        tq.call_count += 1

        output = greedy_decode_with_prefill_cache(
            model=model,
            tokenizer=tok,
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            variant="qmse_packed",
            qmse_bits=tq.bits,
            rotation_seed=tq.rotation_seed,
            num_outlier_channels=tq.num_outlier_channels,
            outlier_extra_bits=tq.outlier_extra_bits,
            use_qjl_keys=tq.use_qjl_keys,
            quantize_decode=tq.quantize_decode,
            norm_guard=tq.norm_guard,
        )

        model._tq_last_output = output

        if not quiet:
            telem = summarize_generation_metrics(output.metrics)
            print(telem.format(compact=True))

        completion_ids = tok.encode(output.text, add_special_tokens=False)
        full_ids = torch.cat([
            input_ids[0],
            torch.tensor(completion_ids, device=input_ids.device),
        ]).unsqueeze(0)
        return full_ids

    model.generate = _tq_generate

    if not quiet:
        model_type = getattr(model.config, "model_type", "unknown")
        num_layers = getattr(model.config, "num_hidden_layers", "?")
        mode = "Q_prod" if use_qjl_keys else "Q_mse"
        guard_status = "on" if norm_guard else "off"
        print()
        print(f"  TurboQuant activated")
        print(f"    Model:      {model_type} ({num_layers} layers)")
        print(f"    Bits:       {bits}-bit {mode}")
        print(f"    Norm guard: {guard_status}")
        print(f"    Decode:     {'quantized' if quantize_decode else 'dense'}")
        print()
        print(f"  model.generate() now uses TurboQuant compression.")
        print(f"  Call turboquant.deactivate(model) to restore original behavior.")
        print()


def deactivate(model, quiet: bool = False) -> None:
    """Deactivate TurboQuant and restore the original ``model.generate()``."""
    state: _TurboQuantState | None = getattr(model, "_tq_state", None)
    if state is None:
        if not quiet:
            print("[TurboQuant] Not active on this model, nothing to deactivate.")
        return

    model.generate = state.original_generate

    for attr in ("_tq_state", "_tq_tokenizer", "_tq_last_output"):
        if hasattr(model, attr):
            delattr(model, attr)

    if not quiet:
        print(f"[TurboQuant] Deactivated after {state.call_count} calls.")


def is_active(model) -> bool:
    """Check whether TurboQuant is currently active on a model."""
    return hasattr(model, "_tq_state")


def last_metrics(model) -> dict[str, Any] | None:
    """Get metrics from the last TurboQuant generate() call."""
    output = getattr(model, "_tq_last_output", None)
    if output is None:
        return None
    return output.metrics.to_dict()


def last_telemetry(model) -> dict[str, Any] | None:
    """Get telemetry summary from the last TurboQuant generate() call as a dict."""
    output = getattr(model, "_tq_last_output", None)
    if output is None:
        return None
    return summarize_generation_metrics(output.metrics).to_dict()


def print_telemetry(model) -> None:
    """Print a formatted telemetry summary from the last generate() call."""
    output = getattr(model, "_tq_last_output", None)
    if output is None:
        print("[TurboQuant] No generation data yet.")
        return
    telem = summarize_generation_metrics(output.metrics)
    print(telem.format(compact=False))
