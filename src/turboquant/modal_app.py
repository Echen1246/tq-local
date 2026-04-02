from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import modal

from turboquant.benchmarks.niah import (
    build_niah_context,
    make_needle_spec,
    niah_system_prompt,
    niah_user_prompt,
    score_niah_response,
)
from turboquant.config import (
    ARTIFACTS_DIR,
    CPU_CORES,
    DEFAULT_ATTN_IMPLEMENTATION,
    DEFAULT_MAX_NEW_TOKENS,
    GPU_TYPE,
    HF_CACHE_DIR,
    MEMORY_MB,
    MODEL_ID,
    MODAL_APP_NAME,
    SUPPORTED_ATTN_IMPLEMENTATIONS,
    resolve_revision,
)
from turboquant.modeling.qwq import QwQLoadConfig, load_qwq_model
from turboquant.prompt_suites import get_prompt_suite
from turboquant.quantization.attention_metrics import causal_attention_logit_mse
from turboquant.quantization.turboquant_mse import (
    quantize_past_key_values_mse,
    quantize_vectors_mse,
    turboquant_mse_analyze,
)
from turboquant.runtime.kv_artifacts import (
    extract_layer_tensor_array,
    extract_layer_vectors,
    load_kv_artifact,
    tensor_map_layers,
)
from turboquant.runtime.kv_capture import (
    model_config_summary,
    save_past_key_values,
    summarize_past_key_values,
)
from turboquant.runtime.experiment_log import log_experiment_event
from turboquant.runtime.metadata import ensure_dir, resolve_run_name, utc_timestamp, write_json
from turboquant.runtime.query_capture import capture_query_projections, save_query_projections

app = modal.App(MODAL_APP_NAME)
hf_cache_volume = modal.Volume.from_name("turboquant-hf-cache", create_if_missing=True)
artifacts_volume = modal.Volume.from_name("turboquant-artifacts", create_if_missing=True)
_local_hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
function_secrets = (
    [
        modal.Secret.from_dict(
            {
                "HF_TOKEN": _local_hf_token,
                "HUGGINGFACE_HUB_TOKEN": _local_hf_token,
            }
        )
    ]
    if _local_hf_token
    else []
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "accelerate==1.13.0",
        "huggingface_hub[hf_transfer]==1.6.0",
        "safetensors==0.7.0",
        "torch==2.10.0",
        "transformers==5.3.0",
    )
    .env(
        {
            "HF_HOME": HF_CACHE_DIR,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONPATH": "/root/src",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_dir(
        "/Users/eddie/Documents/turboquant/src",
        remote_path="/root/src",
        ignore=["__pycache__", "*.pyc"],
    )
)


def _validate_attn_implementation(attn_implementation: str) -> None:
    if attn_implementation not in SUPPORTED_ATTN_IMPLEMENTATIONS:
        raise ValueError(
            f"Unsupported attn_implementation={attn_implementation!r}. "
            f"Choose from {sorted(SUPPORTED_ATTN_IMPLEMENTATIONS)}."
        )


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _parse_int_list(values: str) -> list[int]:
    return [int(item.strip()) for item in values.split(",") if item.strip()]


def _parse_float_list(values: str) -> list[float]:
    return [float(item.strip()) for item in values.split(",") if item.strip()]


def _score_sort_key(item: dict[str, object]) -> tuple[int, float]:
    return (int(item["context_length"]), float(item["depth_percent"]))


def _validate_niah_variant(variant: str) -> None:
    if variant not in {"baseline", "qmse"}:
        raise ValueError(f"Unsupported NIAH variant={variant!r}. Choose from ['baseline', 'qmse'].")


def _eos_token_ids(model, tokenizer) -> set[int]:
    eos_token_id = getattr(model.generation_config, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, int):
        return {eos_token_id}
    return {int(item) for item in eos_token_id}


def _greedy_decode_with_prefill_cache(
    model,
    tokenizer,
    inputs,
    max_new_tokens: int,
    variant: str,
    qmse_bits: int,
    rotation_seed: int,
) -> tuple[str, float, float]:
    import torch

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")

    decode_started = time.monotonic()
    quantization_seconds = 0.0
    eos_token_ids = _eos_token_ids(model, tokenizer)

    with torch.inference_mode():
        outputs = model(**inputs, use_cache=True)
        past_key_values = outputs.past_key_values
        if variant == "qmse":
            quant_started = time.monotonic()
            past_key_values = quantize_past_key_values_mse(
                past_key_values,
                bits=qmse_bits,
                seed=rotation_seed,
            )
            quantization_seconds += time.monotonic() - quant_started

        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens = [next_token]
        attention_mask = inputs.get("attention_mask")

        for _ in range(max_new_tokens - 1):
            if eos_token_ids and int(next_token[0, 0].item()) in eos_token_ids:
                break
            if attention_mask is not None:
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], 1),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ),
                    dim=-1,
                )
            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            if variant == "qmse":
                quant_started = time.monotonic()
                past_key_values = quantize_past_key_values_mse(
                    past_key_values,
                    bits=qmse_bits,
                    seed=rotation_seed,
                    token_slice=slice(-1, None),
                )
                quantization_seconds += time.monotonic() - quant_started
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated_tokens.append(next_token)

    completion_tokens = torch.cat(generated_tokens, dim=-1)
    response_text = tokenizer.decode(completion_tokens[0], skip_special_tokens=True).strip()
    generation_seconds = time.monotonic() - decode_started
    return response_text, generation_seconds, quantization_seconds


def _run_niah_case_impl(
    context_length: int,
    depth_percent: float,
    revision: str | None,
    attn_implementation: str,
    max_new_tokens: int,
    run_name: str | None,
    variant: str,
    qmse_bits: int,
    rotation_seed: int,
) -> dict[str, object]:
    import torch

    _validate_attn_implementation(attn_implementation)
    _validate_niah_variant(variant)
    resolved_revision = resolve_revision(revision)
    case_name = resolve_run_name(
        "niah",
        run_name or f"niah-ctx{context_length}-depth{int(depth_percent)}",
    )
    run_dir = ensure_dir(Path(ARTIFACTS_DIR) / "runs" / case_name)

    tokenizer, model = load_qwq_model(
        QwQLoadConfig(
            revision=resolved_revision,
            token=_hf_token(),
            cache_dir=HF_CACHE_DIR,
            attn_implementation=attn_implementation,
        )
    )

    needle = make_needle_spec(f"{resolved_revision}:{context_length}:{depth_percent}:{case_name}")
    context_text, context_meta = build_niah_context(
        tokenizer=tokenizer,
        context_length=context_length,
        depth_percent=depth_percent,
        needle=needle,
    )
    prompt = niah_user_prompt(context=context_text, needle=needle)
    messages = [
        {"role": "system", "content": niah_system_prompt()},
        {"role": "user", "content": prompt},
    ]
    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered_prompt, return_tensors="pt")
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    if torch.cuda.is_available():
        inputs = {name: tensor.to("cuda") for name, tensor in inputs.items()}

    response_text, generation_seconds, quantization_seconds = _greedy_decode_with_prefill_cache(
        model=model,
        tokenizer=tokenizer,
        inputs=inputs,
        max_new_tokens=max_new_tokens,
        variant=variant,
        qmse_bits=qmse_bits,
        rotation_seed=rotation_seed,
    )
    score = score_niah_response(response_text=response_text, needle=needle)

    metadata = {
        "timestamp_utc": utc_timestamp(),
        "run_name": case_name,
        "benchmark": "niah",
        "variant": variant,
        "model_id": MODEL_ID,
        "revision": resolved_revision,
        "attn_implementation": attn_implementation,
        "context_length": context_length,
        "depth_percent": depth_percent,
        "prompt_tokens": prompt_tokens,
        "generation_seconds": round(generation_seconds, 4),
        "quantization_seconds": round(quantization_seconds, 4),
        "qmse_bits": qmse_bits if variant == "qmse" else None,
        "rotation_seed": rotation_seed if variant == "qmse" else None,
        "decoder_mode": "manual_greedy_prefill_cache",
        "cache_quantization_scope": (
            "prefill_full_cache_and_incremental_generated_tail"
            if variant == "qmse"
            else "none"
        ),
        "needle_key": needle.key,
        "score": score,
        "context_meta": context_meta,
        "model_config": model_config_summary(model),
    }
    metadata_path = run_dir / "niah_result.json"
    response_path = run_dir / "response.txt"
    write_json(metadata_path, metadata)
    response_path.write_text(response_text)
    log_path = log_experiment_event(
        Path(ARTIFACTS_DIR) / "logs",
        event="run_niah_case",
        payload={
            "run_name": case_name,
            "benchmark": "niah",
            "variant": variant,
            "revision": resolved_revision,
            "context_length": context_length,
            "depth_percent": depth_percent,
            "prompt_tokens": prompt_tokens,
            "exact_match": score["exact_match"],
            "generation_seconds": round(generation_seconds, 4),
            "quantization_seconds": round(quantization_seconds, 4),
            "qmse_bits": qmse_bits if variant == "qmse" else None,
            "metadata_path": str(metadata_path),
            "response_path": str(response_path),
        },
    )

    hf_cache_volume.commit()
    artifacts_volume.commit()
    return {
        "run_name": case_name,
        "context_length": context_length,
        "depth_percent": depth_percent,
        "variant": variant,
        "prompt_tokens": prompt_tokens,
        "exact_match": score["exact_match"],
        "response_text": response_text,
        "quantization_seconds": round(quantization_seconds, 4),
        "qmse_bits": qmse_bits if variant == "qmse" else None,
        "metadata_path": str(metadata_path),
        "log_path": str(log_path),
    }


@app.function(
    image=image,
    gpu=GPU_TYPE,
    cpu=CPU_CORES,
    memory=MEMORY_MB,
    timeout=60 * 60,
    volumes={HF_CACHE_DIR: hf_cache_volume, ARTIFACTS_DIR: artifacts_volume},
    secrets=function_secrets,
)
def prefetch_model(revision: str | None = None) -> dict[str, str]:
    from huggingface_hub import snapshot_download

    resolved_revision = resolve_revision(revision)
    print(f"[prefetch] model_id={MODEL_ID} revision={resolved_revision}")
    local_path = snapshot_download(
        repo_id=MODEL_ID,
        revision=resolved_revision,
        token=_hf_token(),
    )
    hf_cache_volume.commit()
    return {
        "model_id": MODEL_ID,
        "revision": resolved_revision,
        "local_path": local_path,
    }


@app.function(
    image=image,
    gpu=GPU_TYPE,
    cpu=CPU_CORES,
    memory=MEMORY_MB,
    timeout=60 * 60,
    volumes={HF_CACHE_DIR: hf_cache_volume, ARTIFACTS_DIR: artifacts_volume},
    secrets=function_secrets,
)
def baseline_generate(
    prompt: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    revision: str | None = None,
    attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION,
    run_name: str | None = None,
) -> dict[str, str | int | float]:
    import torch

    _validate_attn_implementation(attn_implementation)
    resolved_revision = resolve_revision(revision)
    resolved_run_name = resolve_run_name("baseline", run_name)
    run_dir = ensure_dir(Path(ARTIFACTS_DIR) / "runs" / resolved_run_name)

    print(f"[baseline] run_name={resolved_run_name} revision={resolved_revision}")
    load_started = time.monotonic()
    tokenizer, model = load_qwq_model(
        QwQLoadConfig(
            revision=resolved_revision,
            token=_hf_token(),
            cache_dir=HF_CACHE_DIR,
            attn_implementation=attn_implementation,
        )
    )
    load_seconds = time.monotonic() - load_started
    print(f"[baseline] model loaded in {load_seconds:.2f}s")

    messages = [{"role": "user", "content": prompt}]
    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered_prompt, return_tensors="pt")
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    if torch.cuda.is_available():
        inputs = {name: tensor.to("cuda") for name, tensor in inputs.items()}

    print(f"[baseline] prompt_tokens={prompt_tokens}")
    generation_started = time.monotonic()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    generation_seconds = time.monotonic() - generation_started
    print(f"[baseline] generation finished in {generation_seconds:.2f}s")

    completion_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(completion_tokens, skip_special_tokens=True)

    metadata = {
        "timestamp_utc": utc_timestamp(),
        "run_name": resolved_run_name,
        "model_id": MODEL_ID,
        "revision": resolved_revision,
        "attn_implementation": attn_implementation,
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": int(completion_tokens.shape[-1]),
        "prompt_chars": len(prompt),
        "completion_chars": len(text),
        "load_seconds": round(load_seconds, 4),
        "generation_seconds": round(generation_seconds, 4),
        "model_config": model_config_summary(model),
    }
    metadata_path = run_dir / "baseline_metadata.json"
    response_path = run_dir / "response.txt"
    write_json(metadata_path, metadata)
    response_path.write_text(text)
    log_path = log_experiment_event(
        Path(ARTIFACTS_DIR) / "logs",
        event="baseline_generate",
        payload={
            "run_name": resolved_run_name,
            "model_id": MODEL_ID,
            "revision": resolved_revision,
            "attn_implementation": attn_implementation,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": int(completion_tokens.shape[-1]),
            "load_seconds": round(load_seconds, 4),
            "generation_seconds": round(generation_seconds, 4),
            "metadata_path": str(metadata_path),
            "response_path": str(response_path),
        },
    )

    hf_cache_volume.commit()
    artifacts_volume.commit()
    return {
        "run_name": resolved_run_name,
        "model_id": MODEL_ID,
        "revision": resolved_revision,
        "attn_implementation": attn_implementation,
        "prompt_chars": len(prompt),
        "completion_chars": len(text),
        "prompt_tokens": prompt_tokens,
        "load_seconds": round(load_seconds, 4),
        "generation_seconds": round(generation_seconds, 4),
        "metadata_path": str(metadata_path),
        "response_path": str(response_path),
        "log_path": str(log_path),
        "text": text,
    }


@app.function(
    image=image,
    gpu=GPU_TYPE,
    cpu=CPU_CORES,
    memory=MEMORY_MB,
    timeout=60 * 60,
    volumes={HF_CACHE_DIR: hf_cache_volume, ARTIFACTS_DIR: artifacts_volume},
    secrets=function_secrets,
)
def capture_prompt_kv(
    prompt: str,
    revision: str | None = None,
    attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION,
    run_name: str | None = None,
) -> dict[str, str | int]:
    import torch

    _validate_attn_implementation(attn_implementation)
    resolved_revision = resolve_revision(revision)
    resolved_run_name = resolve_run_name("capture-kv", run_name)
    run_dir = ensure_dir(Path(ARTIFACTS_DIR) / "runs" / resolved_run_name)

    print(f"[capture-kv] run_name={resolved_run_name} revision={resolved_revision}")
    tokenizer, model = load_qwq_model(
        QwQLoadConfig(
            revision=resolved_revision,
            token=_hf_token(),
            cache_dir=HF_CACHE_DIR,
            attn_implementation=attn_implementation,
        )
    )
    messages = [{"role": "user", "content": prompt}]
    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered_prompt, return_tensors="pt")
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    if torch.cuda.is_available():
        inputs = {name: tensor.to("cuda") for name, tensor in inputs.items()}

    print(f"[capture-kv] prompt_tokens={prompt_tokens}")
    with torch.inference_mode():
        query_state_dict, outputs = capture_query_projections(
            model,
            lambda: model(**inputs, use_cache=True),
        )

    tensor_path = run_dir / "prompt_kv.safetensors"
    query_path = run_dir / "prompt_queries.safetensors"
    metadata_path = run_dir / "kv_metadata.json"
    save_past_key_values(tensor_path, outputs.past_key_values)
    save_query_projections(query_path, query_state_dict)
    metadata = {
        "timestamp_utc": utc_timestamp(),
        "run_name": resolved_run_name,
        "model_id": MODEL_ID,
        "revision": resolved_revision,
        "attn_implementation": attn_implementation,
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "tensor_path": str(tensor_path),
        "query_path": str(query_path),
        "model_config": model_config_summary(model),
        "layer_summaries": summarize_past_key_values(outputs.past_key_values),
    }
    write_json(metadata_path, metadata)
    log_path = log_experiment_event(
        Path(ARTIFACTS_DIR) / "logs",
        event="capture_prompt_kv",
        payload={
            "run_name": resolved_run_name,
            "model_id": MODEL_ID,
            "revision": resolved_revision,
            "attn_implementation": attn_implementation,
            "prompt_tokens": prompt_tokens,
            "tensor_path": str(tensor_path),
            "query_path": str(query_path),
            "metadata_path": str(metadata_path),
        },
    )

    hf_cache_volume.commit()
    artifacts_volume.commit()
    return {
        "run_name": resolved_run_name,
        "model_id": MODEL_ID,
        "revision": resolved_revision,
        "prompt_tokens": prompt_tokens,
        "metadata_path": str(metadata_path),
        "tensor_path": str(tensor_path),
        "query_path": str(query_path),
        "log_path": str(log_path),
    }


@app.function(
    image=image,
    cpu=1,
    memory=1024,
    timeout=10 * 60,
    volumes={ARTIFACTS_DIR: artifacts_volume},
)
def inspect_run_artifacts(run_name: str) -> dict:
    run_dir = Path(ARTIFACTS_DIR) / "runs" / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Run artifacts not found for run_name={run_name!r}")

    result: dict[str, object] = {"run_name": run_name}

    kv_metadata_path = run_dir / "kv_metadata.json"
    if kv_metadata_path.exists():
        kv_metadata = _read_json(kv_metadata_path)
        layer_summaries = kv_metadata.get("layer_summaries", [])
        key_means = [layer["key_mean_token_norm"] for layer in layer_summaries]
        value_means = [layer["value_mean_token_norm"] for layer in layer_summaries]
        key_stds = [layer["key_std_token_norm"] for layer in layer_summaries]
        value_stds = [layer["value_std_token_norm"] for layer in layer_summaries]
        result["kv_metadata"] = kv_metadata
        result["kv_summary"] = {
            "num_layers": len(layer_summaries),
            "prompt_tokens": kv_metadata.get("prompt_tokens"),
            "tensor_path": kv_metadata.get("tensor_path"),
            "model_config": kv_metadata.get("model_config"),
            "first_layer": layer_summaries[0] if layer_summaries else None,
            "last_layer": layer_summaries[-1] if layer_summaries else None,
            "key_mean_range": [min(key_means), max(key_means)] if key_means else None,
            "value_mean_range": [min(value_means), max(value_means)] if value_means else None,
            "key_std_range": [min(key_stds), max(key_stds)] if key_stds else None,
            "value_std_range": [min(value_stds), max(value_stds)] if value_stds else None,
            "mid_layers": (
                [layer_summaries[len(layer_summaries) // 2 - 1], layer_summaries[len(layer_summaries) // 2]]
                if len(layer_summaries) >= 2
                else layer_summaries
            ),
        }
        result["kv_metadata"] = {
            "timestamp_utc": kv_metadata.get("timestamp_utc"),
            "run_name": kv_metadata.get("run_name"),
            "model_id": kv_metadata.get("model_id"),
            "revision": kv_metadata.get("revision"),
            "attn_implementation": kv_metadata.get("attn_implementation"),
            "prompt": kv_metadata.get("prompt"),
            "prompt_tokens": kv_metadata.get("prompt_tokens"),
            "tensor_path": kv_metadata.get("tensor_path"),
            "model_config": kv_metadata.get("model_config"),
        }

    baseline_metadata_path = run_dir / "baseline_metadata.json"
    if baseline_metadata_path.exists():
        result["baseline_metadata"] = _read_json(baseline_metadata_path)

    response_path = run_dir / "response.txt"
    if response_path.exists():
        result["response_text"] = response_path.read_text()

    return result


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=60 * 60,
    volumes={ARTIFACTS_DIR: artifacts_volume},
)
def analyze_turboquant_mse_run(
    run_name: str,
    bits: int = 3,
    bits_list: str | None = None,
    rotation_seed: int = 0,
    query_seed: int = 0,
    num_query_samples: int = 128,
    target: str = "both",
) -> dict:
    if target not in {"key", "value", "both"}:
        raise ValueError(f"Unsupported target={target!r}")

    run_dir = Path(ARTIFACTS_DIR) / "runs" / run_name
    kv_metadata_path = run_dir / "kv_metadata.json"
    if not kv_metadata_path.exists():
        raise FileNotFoundError(f"Missing KV metadata for run_name={run_name!r}")

    kv_metadata = _read_json(kv_metadata_path)
    tensor_map = load_kv_artifact(kv_metadata["tensor_path"])
    query_tensor_map = (
        load_kv_artifact(kv_metadata["query_path"])
        if kv_metadata.get("query_path")
        else None
    )
    layers = tensor_map_layers(tensor_map)
    kinds = ["key", "value"] if target == "both" else [target]
    bit_values = (
        [int(item.strip()) for item in bits_list.split(",") if item.strip()]
        if bits_list
        else [bits]
    )

    per_bit: dict[str, dict[str, list[dict]]] = {}
    for bit_value in bit_values:
        per_kind: dict[str, list[dict]] = {kind: [] for kind in kinds}
        for kind in kinds:
            for layer_index in layers:
                vectors = extract_layer_vectors(tensor_map, layer_index=layer_index, kind=kind)
                metrics = turboquant_mse_analyze(
                    vectors=vectors,
                    bits=bit_value,
                    seed=rotation_seed,
                    query_seed=query_seed + layer_index,
                    num_query_samples=num_query_samples,
                )
                metrics["kind"] = kind
                metrics["layer"] = layer_index
                if kind == "key" and query_tensor_map is not None:
                    original_keys = extract_layer_tensor_array(tensor_map, layer_index=layer_index, kind="key")
                    reconstructed_keys, _ = quantize_vectors_mse(
                        vectors=original_keys.reshape(-1, original_keys.shape[-1]),
                        bits=bit_value,
                        seed=rotation_seed,
                    )
                    reconstructed_keys = reconstructed_keys.reshape(original_keys.shape)
                    queries = extract_layer_tensor_array(
                        query_tensor_map,
                        layer_index=layer_index,
                        kind="query",
                    )
                    metrics["actual_query_causal_logit_mse"] = causal_attention_logit_mse(
                        queries=queries,
                        keys=original_keys,
                        reconstructed_keys=reconstructed_keys,
                    )
                per_kind[kind].append(metrics)
        per_bit[str(bit_value)] = per_kind

    def aggregate(layer_metrics: list[dict]) -> dict:
        aggregate_payload = {
            "num_layers": len(layer_metrics),
            "mean_mse": sum(item["mse"] for item in layer_metrics) / max(len(layer_metrics), 1),
            "mean_cosine": sum(item["mean_cosine"] for item in layer_metrics) / max(len(layer_metrics), 1),
            "mean_inner_product_mse": (
                sum(item["inner_product_mse"] for item in layer_metrics) / max(len(layer_metrics), 1)
            ),
            "best_layer_by_cosine": max(layer_metrics, key=lambda item: item["mean_cosine"]),
            "worst_layer_by_cosine": min(layer_metrics, key=lambda item: item["mean_cosine"]),
        }
        if any("actual_query_causal_logit_mse" in item for item in layer_metrics):
            query_items = [item["actual_query_causal_logit_mse"] for item in layer_metrics]
            aggregate_payload["mean_actual_query_causal_logit_mse"] = sum(query_items) / len(query_items)
            aggregate_payload["inner_product_metric_type"] = "actual_model_query_causal_logit_mse"
        return aggregate_payload

    aggregate_by_bit = {
        bit_key: {kind: aggregate(per_bit[bit_key][kind]) for kind in kinds} for bit_key in per_bit
    }

    summary = {
        "analysis": "turboquant_mse",
        "run_name": run_name,
        "source_tensor_path": kv_metadata["tensor_path"],
        "revision": kv_metadata["revision"],
        "bits": bits,
        "bits_list": bit_values,
        "rotation_seed": rotation_seed,
        "query_seed": query_seed,
        "num_query_samples": num_query_samples,
        "target": target,
        "model_config": kv_metadata["model_config"],
        "aggregate_by_bit": aggregate_by_bit,
        "per_bit_layers": per_bit,
    }

    analysis_dir = ensure_dir(run_dir / "analysis")
    if len(bit_values) == 1:
        analysis_filename = f"turboquant_mse_b{bit_values[0]}_{target}.json"
    else:
        joined = "-".join(str(item) for item in bit_values)
        analysis_filename = f"turboquant_mse_sweep_b{joined}_{target}.json"
    analysis_path = analysis_dir / analysis_filename
    write_json(analysis_path, summary)
    log_path = log_experiment_event(
        Path(ARTIFACTS_DIR) / "logs",
        event="analyze_turboquant_mse_run",
        payload={
            "run_name": run_name,
            "revision": kv_metadata["revision"],
            "bits": bits,
            "bits_list": bit_values,
            "target": target,
            "analysis_path": str(analysis_path),
            "inner_product_metric_type": (
                "actual_model_query_causal_logit_mse"
                if query_tensor_map is not None
                else "random_unit_query_proxy"
            ),
            "aggregate_by_bit": summary["aggregate_by_bit"],
        },
    )
    artifacts_volume.commit()
    return {
        "run_name": run_name,
        "analysis_path": str(analysis_path),
        "log_path": str(log_path),
        "summary": {
            "bits": bits,
            "bits_list": bit_values,
            "target": target,
            "aggregate_by_bit": summary["aggregate_by_bit"],
        },
    }


@app.function(
    image=image,
    gpu=GPU_TYPE,
    cpu=CPU_CORES,
    memory=MEMORY_MB,
    timeout=60 * 60,
    volumes={HF_CACHE_DIR: hf_cache_volume, ARTIFACTS_DIR: artifacts_volume},
    secrets=function_secrets,
)
def run_niah_case(
    context_length: int,
    depth_percent: float,
    revision: str | None = None,
    attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION,
    max_new_tokens: int = 32,
    run_name: str | None = None,
    variant: str = "baseline",
    qmse_bits: int = 3,
    rotation_seed: int = 0,
) -> dict[str, object]:
    return _run_niah_case_impl(
        context_length=context_length,
        depth_percent=depth_percent,
        revision=revision,
        attn_implementation=attn_implementation,
        max_new_tokens=max_new_tokens,
        run_name=run_name,
        variant=variant,
        qmse_bits=qmse_bits,
        rotation_seed=rotation_seed,
    )


@app.function(
    image=image,
    gpu=GPU_TYPE,
    cpu=CPU_CORES,
    memory=MEMORY_MB,
    timeout=60 * 60 * 4,
    volumes={HF_CACHE_DIR: hf_cache_volume, ARTIFACTS_DIR: artifacts_volume},
    secrets=function_secrets,
)
def run_niah_grid(
    context_lengths: str = "4000,8000,16000,32000",
    depth_percents: str = "10,50,90",
    revision: str | None = None,
    attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION,
    max_new_tokens: int = 32,
    run_name: str | None = None,
    variant: str = "baseline",
    qmse_bits: int = 3,
    rotation_seed: int = 0,
) -> dict[str, object]:
    lengths = _parse_int_list(context_lengths)
    depths = _parse_float_list(depth_percents)
    grid_name = resolve_run_name("niah-grid", run_name)
    results = []

    for context_length in lengths:
        for depth_percent in depths:
            case_result = _run_niah_case_impl(
                context_length=context_length,
                depth_percent=depth_percent,
                revision=revision,
                attn_implementation=attn_implementation,
                max_new_tokens=max_new_tokens,
                run_name=f"{grid_name}-ctx{context_length}-d{int(depth_percent)}",
                variant=variant,
                qmse_bits=qmse_bits,
                rotation_seed=rotation_seed,
            )
            results.append(case_result)

    summary = {
        "run_name": grid_name,
        "benchmark": "niah",
        "variant": variant,
        "qmse_bits": qmse_bits if variant == "qmse" else None,
        "rotation_seed": rotation_seed if variant == "qmse" else None,
        "num_cases": len(results),
        "context_lengths": lengths,
        "depth_percents": depths,
        "exact_match_rate": sum(1 for item in results if item["exact_match"]) / max(len(results), 1),
        "results": results,
    }
    summary_dir = ensure_dir(Path(ARTIFACTS_DIR) / "runs" / grid_name)
    summary_path = summary_dir / "niah_grid_summary.json"
    write_json(summary_path, summary)
    log_path = log_experiment_event(
        Path(ARTIFACTS_DIR) / "logs",
        event="run_niah_grid",
        payload={
            "run_name": grid_name,
            "benchmark": "niah",
            "variant": variant,
            "qmse_bits": qmse_bits if variant == "qmse" else None,
            "rotation_seed": rotation_seed if variant == "qmse" else None,
            "context_lengths": lengths,
            "depth_percents": depths,
            "num_cases": len(results),
            "exact_match_rate": summary["exact_match_rate"],
            "summary_path": str(summary_path),
        },
    )
    artifacts_volume.commit()
    return {
        "run_name": grid_name,
        "summary_path": str(summary_path),
        "log_path": str(log_path),
        "exact_match_rate": summary["exact_match_rate"],
        "num_cases": len(results),
    }


@app.function(
    image=image,
    cpu=1,
    memory=1024,
    timeout=10 * 60,
    volumes={ARTIFACTS_DIR: artifacts_volume},
)
def compare_niah_grids(baseline_run_name: str, candidate_run_name: str) -> dict[str, object]:
    baseline_path = Path(ARTIFACTS_DIR) / "runs" / baseline_run_name / "niah_grid_summary.json"
    candidate_path = Path(ARTIFACTS_DIR) / "runs" / candidate_run_name / "niah_grid_summary.json"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline NIAH summary for run {baseline_run_name!r}")
    if not candidate_path.exists():
        raise FileNotFoundError(f"Missing candidate NIAH summary for run {candidate_run_name!r}")

    baseline = _read_json(baseline_path)
    candidate = _read_json(candidate_path)
    baseline_results = sorted(baseline["results"], key=_score_sort_key)
    candidate_results = sorted(candidate["results"], key=_score_sort_key)
    if len(baseline_results) != len(candidate_results):
        raise ValueError("Baseline and candidate NIAH grids have different case counts.")

    comparisons = []
    for left, right in zip(baseline_results, candidate_results):
        if _score_sort_key(left) != _score_sort_key(right):
            raise ValueError("Baseline and candidate NIAH grids do not align by context/depth.")
        comparisons.append(
            {
                "context_length": left["context_length"],
                "depth_percent": left["depth_percent"],
                "baseline_exact_match": left["exact_match"],
                "candidate_exact_match": right["exact_match"],
                "match_delta": int(bool(right["exact_match"])) - int(bool(left["exact_match"])),
            }
        )

    return {
        "baseline_run_name": baseline_run_name,
        "candidate_run_name": candidate_run_name,
        "baseline_variant": baseline.get("variant"),
        "candidate_variant": candidate.get("variant"),
        "baseline_exact_match_rate": baseline["exact_match_rate"],
        "candidate_exact_match_rate": candidate["exact_match_rate"],
        "comparisons": comparisons,
    }


@app.local_entrypoint()
def main(
    prompt: str = "Say hello in one sentence.",
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    revision: str | None = None,
    attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION,
    prefetch_only: bool = False,
    capture_kv: bool = False,
    inspect_run: str | None = None,
    analyze_turboquant_mse: str | None = None,
    bits: int = 3,
    bits_list: str | None = None,
    target: str = "both",
    capture_suite: str | None = None,
    niah_context_length: int | None = None,
    niah_depth_percent: float = 50.0,
    niah_grid: bool = False,
    compare_niah_baseline: str | None = None,
    compare_niah_candidate: str | None = None,
    context_lengths: str = "4000,8000,16000,32000",
    depth_percents: str = "10,50,90",
    variant: str = "baseline",
    qmse_bits: int = 3,
    rotation_seed: int = 0,
    run_name: str | None = None,
) -> None:
    if prefetch_only:
        result = prefetch_model.remote(revision=revision)
        print(result["local_path"])
        return

    if inspect_run:
        result = inspect_run_artifacts.remote(run_name=inspect_run)
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if analyze_turboquant_mse:
        result = analyze_turboquant_mse_run.remote(
            run_name=analyze_turboquant_mse,
            bits=bits,
            bits_list=bits_list,
            target=target,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if capture_suite:
        suite = get_prompt_suite(capture_suite)
        results = []
        for index, suite_prompt in enumerate(suite):
            suite_run_name = f"{capture_suite}-{index:02d}"
            result = capture_prompt_kv.remote(
                prompt=suite_prompt,
                revision=revision,
                attn_implementation=attn_implementation,
                run_name=suite_run_name,
            )
            results.append(result)
        print(json.dumps(results, indent=2, sort_keys=True))
        return

    if compare_niah_baseline and compare_niah_candidate:
        result = compare_niah_grids.remote(
            baseline_run_name=compare_niah_baseline,
            candidate_run_name=compare_niah_candidate,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if niah_grid:
        result = run_niah_grid.remote(
            context_lengths=context_lengths,
            depth_percents=depth_percents,
            revision=revision,
            attn_implementation=attn_implementation,
            max_new_tokens=max_new_tokens,
            variant=variant,
            qmse_bits=qmse_bits,
            rotation_seed=rotation_seed,
            run_name=run_name,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if niah_context_length is not None:
        result = run_niah_case.remote(
            context_length=niah_context_length,
            depth_percent=niah_depth_percent,
            revision=revision,
            attn_implementation=attn_implementation,
            max_new_tokens=max_new_tokens,
            variant=variant,
            qmse_bits=qmse_bits,
            rotation_seed=rotation_seed,
            run_name=run_name,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if capture_kv:
        result = capture_prompt_kv.remote(
            prompt=prompt,
            revision=revision,
            attn_implementation=attn_implementation,
            run_name=run_name,
        )
        print(result["metadata_path"])
        return

    result = baseline_generate.remote(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        revision=revision,
        attn_implementation=attn_implementation,
        run_name=run_name,
    )
    print(result["text"])
