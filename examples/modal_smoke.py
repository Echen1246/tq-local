from __future__ import annotations

import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import modal

from turboquant import TurboQuantSession

app = modal.App("turboquant-smoke")
HF_CACHE_DIR = "/vol/hf-cache"
hf_cache_volume = modal.Volume.from_name("tq-local-hf-cache", create_if_missing=True)
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
    .pip_install(
        "accelerate==1.13.0",
        "huggingface_hub[hf_transfer]==1.6.0",
        "numpy==2.4.3",
        "safetensors==0.7.0",
        "scipy==1.17.1",
        "torch==2.10.0",
        "transformers==5.3.0",
    )
    .env(
        {
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": HF_CACHE_DIR,
            "PYTHONPATH": "/root/src",
        }
    )
    .add_local_dir(
        str(Path(__file__).resolve().parents[1] / "src"),
        remote_path="/root/src",
        ignore=["__pycache__", "*.pyc"],
    )
)


@app.cls(
    image=image,
    gpu="B200",
    cpu=4,
    memory=16384,
    timeout=30 * 60,
    secrets=function_secrets,
    volumes={HF_CACHE_DIR: hf_cache_volume},
    scaledown_window=20 * 60,
    min_containers=1,
)
class SmokeRunner:
    @modal.enter()
    def load(self) -> None:
        self._sessions: dict[tuple, TurboQuantSession] = {}

    def _session_for(
        self,
        model: str,
        variant: str,
        bits: int,
        num_outlier_channels: int = 0,
        outlier_extra_bits: int = 1,
        use_qjl_keys: bool = False,
        quantize_decode: bool = False,
        norm_guard: bool = True,
    ) -> TurboQuantSession:
        key = (model, variant, bits, num_outlier_channels, outlier_extra_bits,
               use_qjl_keys, quantize_decode, norm_guard)
        session = self._sessions.get(key)
        if session is None:
            session = TurboQuantSession.from_pretrained(
                model,
                variant=variant,
                bits=bits,
                dtype="auto",
                device_map="auto",
                token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
                cache_dir=HF_CACHE_DIR,
                num_outlier_channels=num_outlier_channels,
                outlier_extra_bits=outlier_extra_bits,
                use_qjl_keys=use_qjl_keys,
                quantize_decode=quantize_decode,
                norm_guard=norm_guard,
            )
            self._sessions[key] = session
            hf_cache_volume.commit()
        return session

    @modal.method()
    def run(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        variant: str = "qmse_packed",
        bits: int = 3,
        prompt: str = "Explain KV cache compression in one short paragraph.",
        max_new_tokens: int = 128,
        num_outlier_channels: int = 0,
        outlier_extra_bits: int = 1,
        use_qjl_keys: bool = False,
        quantize_decode: bool = False,
        norm_guard: bool = True,
    ) -> dict[str, object]:
        session = self._session_for(
            model=model,
            variant=variant,
            bits=bits,
            num_outlier_channels=num_outlier_channels,
            outlier_extra_bits=outlier_extra_bits,
            use_qjl_keys=use_qjl_keys,
            quantize_decode=quantize_decode,
            norm_guard=norm_guard,
        )
        output = session.generate(prompt=prompt, max_new_tokens=max_new_tokens, return_output=True)
        text = output.text
        effective_bits = bits
        if variant != "baseline" and num_outlier_channels > 0:
            cfg = session.model.config
            head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
            normal_channels = head_dim - num_outlier_channels
            effective_bits = (
                num_outlier_channels * (bits + outlier_extra_bits) + normal_channels * bits
            ) / head_dim
        recon = output.metrics.reconstruction_quality
        recon_summary = None
        if recon:
            quantized = [r for r in recon if not r.get("dense", False)]
            dense_layers = [r["layer"] for r in recon if r.get("dense", False)]
            src = quantized if quantized else recon
            avg_key_cos = sum(r["key_cosine_sim"] for r in src) / len(src)
            avg_val_cos = sum(r["val_cosine_sim"] for r in src) / len(src)
            avg_key_mse = sum(r["key_mse"] for r in src) / len(src)
            avg_val_mse = sum(r["val_mse"] for r in src) / len(src)
            worst_key_mse_layer = max(src, key=lambda r: r["key_mse"])
            max_key_norm = max(r["key_mean_norm"] for r in recon)
            recon_summary = {
                "layers_quantized": len(quantized),
                "layers_dense": len(dense_layers),
                "dense_layer_ids": dense_layers,
                "avg_key_cosine_sim": round(avg_key_cos, 6),
                "avg_val_cosine_sim": round(avg_val_cos, 6),
                "avg_key_mse": round(avg_key_mse, 6),
                "avg_val_mse": round(avg_val_mse, 6),
                "max_key_mean_norm": round(max_key_norm, 2),
                "worst_key_mse_layer": worst_key_mse_layer,
            }

        return {
            "model": model,
            "variant": variant,
            "bits": bits if variant != "baseline" else None,
            "num_outlier_channels": num_outlier_channels if variant != "baseline" else None,
            "outlier_extra_bits": outlier_extra_bits if variant != "baseline" else None,
            "use_qjl_keys": use_qjl_keys if variant != "baseline" else None,
            "quantize_decode": quantize_decode if variant != "baseline" else None,
            "norm_guard": norm_guard if variant != "baseline" else None,
            "effective_bits": round(effective_bits, 3) if variant != "baseline" else None,
            "text": text,
            "reconstruction_quality": recon_summary,
            "telemetry": session.last_telemetry(),
            "compatibility": session.compatibility_report(),
        }


    @modal.method()
    def memory_benchmark(
        self,
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        bits: int = 3,
        prompt_tokens: int = 2048,
        max_new_tokens: int = 16,
        use_qjl_keys: bool = False,
        quantize_decode: bool = False,
    ) -> dict[str, object]:
        """Run baseline vs TurboQuant and compare peak VRAM.

        Measures incremental memory above model weights (peak_minus_pre)
        to isolate KV cache + attention overhead from static model size.
        """
        import gc
        import torch

        sentence = "The quick brown fox jumps over the lazy dog. "
        repeats = max(500, (prompt_tokens * 5) // len(sentence) + 1)
        filler = sentence * repeats
        prompt = filler[:prompt_tokens * 5]

        results = {}
        for label, variant, qjl, qdec in [
            ("baseline", "baseline", False, False),
            ("turboquant", "qmse_packed", use_qjl_keys, quantize_decode),
        ]:
            self._sessions.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            session = self._session_for(
                model=model, variant=variant, bits=bits,
                use_qjl_keys=qjl, quantize_decode=qdec,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            pre_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            output = session.generate(
                prompt=prompt, max_new_tokens=max_new_tokens, return_output=True,
            )
            post_peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            telem = session.last_telemetry()
            results[label] = {
                "pre_allocated_bytes": pre_mem,
                "peak_allocated_bytes": post_peak,
                "peak_minus_pre_bytes": post_peak - pre_mem,
                "prompt_tokens": output.metrics.prompt_tokens,
                "completion_tokens": output.metrics.completion_tokens,
                "generation_seconds": output.metrics.generation_seconds,
                "packed_actual_bytes": telem.get("packed_actual_bytes") if telem else None,
                "dense_kv_bytes": telem.get("dense_kv_bytes") if telem else None,
                "text_preview": output.text[:100],
            }

        baseline_incr = results["baseline"]["peak_minus_pre_bytes"]
        tq_incr = results["turboquant"]["peak_minus_pre_bytes"]
        return {
            "model": model,
            "bits": bits,
            "use_qjl_keys": use_qjl_keys,
            "quantize_decode": quantize_decode,
            "baseline": results["baseline"],
            "turboquant": results["turboquant"],
            "kv_overhead_savings_bytes": baseline_incr - tq_incr,
            "kv_overhead_savings_percent": round(
                (baseline_incr - tq_incr) / baseline_incr * 100, 2
            ) if baseline_incr > 0 else 0,
        }

    @modal.method()
    def profile_channels(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        prompt: str = "Explain KV cache compression in one short paragraph.",
    ) -> dict[str, object]:
        """Profile per-channel key energy to determine if extreme norms are concentrated or distributed."""
        import torch

        session = self._session_for(model=model, variant="baseline", bits=3)
        inputs = session.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = session.model(**inputs, use_cache=True)
            past_kv = outputs.past_key_values

        layer_profiles = []
        for layer_idx, (key_states, val_states, *_) in enumerate(past_kv):
            k = key_states.float()
            channel_energy = k.pow(2).sum(dim=(0, 1, 2))  # [head_dim]
            total_energy = channel_energy.sum().item()

            sorted_energy, sorted_idx = channel_energy.sort(descending=True)
            cumulative = sorted_energy.cumsum(0) / total_energy

            n_50 = int((cumulative < 0.5).sum().item()) + 1
            n_90 = int((cumulative < 0.9).sum().item()) + 1
            n_99 = int((cumulative < 0.99).sum().item()) + 1

            top5_idx = sorted_idx[:5].tolist()
            top5_pct = [round(sorted_energy[i].item() / total_energy * 100, 2) for i in range(5)]

            mean_norm = k.norm(dim=-1).mean().item()
            max_channel = sorted_energy[0].item()
            min_channel = sorted_energy[-1].item()
            ratio = max_channel / min_channel if min_channel > 0 else float("inf")

            layer_profiles.append({
                "layer": layer_idx,
                "key_mean_norm": round(mean_norm, 2),
                "channels_for_50pct_energy": n_50,
                "channels_for_90pct_energy": n_90,
                "channels_for_99pct_energy": n_99,
                "top5_channels": top5_idx,
                "top5_pct_of_total": top5_pct,
                "max_min_channel_ratio": round(ratio, 1),
            })

        flagged = [lp for lp in layer_profiles if lp["key_mean_norm"] > 50]
        return {
            "model": model,
            "head_dim": int(key_states.shape[-1]),
            "num_layers": len(layer_profiles),
            "flagged_layers": flagged,
            "normal_layer_example": layer_profiles[10] if len(layer_profiles) > 10 else layer_profiles[-1],
        }

    @modal.method()
    def output_test(
        self,
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        bits: int = 4,
        max_new_tokens: int = 64,
        use_qjl_keys: bool = False,
        quantize_decode: bool = False,
    ) -> dict[str, object]:
        """Compare baseline vs TurboQuant outputs on fixed prompts.

        Tests:
        1. Quality gates: cosine sim, MSE within expected bounds
        2. Prefix overlap: how many leading tokens match before divergence
        3. Reproducibility: two TurboQuant runs produce identical output
        """
        import gc

        import torch

        test_prompts = [
            "What is 2 + 2? Answer with just the number.",
            "List the first 5 prime numbers separated by commas.",
            "Translate 'hello world' to French in one phrase.",
        ]

        quality_thresholds = {
            4: {"min_cosine": 0.990, "max_key_mse": 0.10},
            3: {"min_cosine": 0.970, "max_key_mse": 0.50},
            2: {"min_cosine": 0.900, "max_key_mse": 2.00},
        }
        thresholds = quality_thresholds.get(bits, quality_thresholds[3])

        results = []
        all_passed = True

        for prompt in test_prompts:
            self._sessions.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            baseline_session = self._session_for(
                model=model, variant="baseline", bits=bits,
            )
            baseline_out = baseline_session.generate(
                prompt=prompt, max_new_tokens=max_new_tokens, return_output=True,
            )
            baseline_text = baseline_out.text
            baseline_tokens = baseline_out.metrics.completion_tokens

            self._sessions.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            tq_session = self._session_for(
                model=model, variant="qmse_packed", bits=bits,
                use_qjl_keys=use_qjl_keys, quantize_decode=quantize_decode,
            )
            tq_out = tq_session.generate(
                prompt=prompt, max_new_tokens=max_new_tokens, return_output=True,
            )
            tq_text = tq_out.text
            tq_tokens = tq_out.metrics.completion_tokens

            tq_out_2 = tq_session.generate(
                prompt=prompt, max_new_tokens=max_new_tokens, return_output=True,
            )
            tq_text_2 = tq_out_2.text
            reproducible = tq_text == tq_text_2

            recon = tq_out.metrics.reconstruction_quality
            avg_key_cos = None
            avg_val_cos = None
            avg_key_mse = None
            if recon:
                quantized = [r for r in recon if not r.get("dense", False)]
                src = quantized if quantized else recon
                avg_key_cos = sum(r["key_cosine_sim"] for r in src) / len(src)
                avg_val_cos = sum(r["val_cosine_sim"] for r in src) / len(src)
                avg_key_mse = sum(r["key_mse"] for r in src) / len(src)

            cosine_pass = avg_key_cos is not None and avg_key_cos >= thresholds["min_cosine"]
            mse_pass = avg_key_mse is not None and avg_key_mse <= thresholds["max_key_mse"]

            baseline_words = baseline_text.split()
            tq_words = tq_text.split()
            prefix_match = 0
            for bw, tw in zip(baseline_words, tq_words):
                if bw == tw:
                    prefix_match += 1
                else:
                    break

            test_passed = cosine_pass and mse_pass and reproducible
            if not test_passed:
                all_passed = False

            results.append({
                "prompt": prompt,
                "passed": test_passed,
                "baseline_text": baseline_text,
                "turboquant_text": tq_text,
                "baseline_tokens": baseline_tokens,
                "turboquant_tokens": tq_tokens,
                "prefix_words_match": prefix_match,
                "total_baseline_words": len(baseline_words),
                "reproducible": reproducible,
                "avg_key_cosine_sim": round(avg_key_cos, 6) if avg_key_cos else None,
                "avg_val_cosine_sim": round(avg_val_cos, 6) if avg_val_cos else None,
                "avg_key_mse": round(avg_key_mse, 6) if avg_key_mse else None,
                "cosine_threshold": thresholds["min_cosine"],
                "cosine_pass": cosine_pass,
                "mse_threshold": thresholds["max_key_mse"],
                "mse_pass": mse_pass,
            })

        return {
            "model": model,
            "bits": bits,
            "use_qjl_keys": use_qjl_keys,
            "quantize_decode": quantize_decode,
            "all_passed": all_passed,
            "tests_run": len(results),
            "tests_passed": sum(1 for r in results if r["passed"]),
            "results": results,
        }


    @modal.method()
    def activate_test(
        self,
        model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        bits: int = 4,
    ) -> dict[str, object]:
        """Test the turboquant.activate() / deactivate() API."""
        import turboquant
        from transformers import AutoModelForCausalLM, AutoTokenizer

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=token, cache_dir=HF_CACHE_DIR,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, token=token, cache_dir=HF_CACHE_DIR,
            device_map="auto", torch_dtype="auto",
        )
        model.eval()

        turboquant.activate(model, tokenizer, bits=bits)
        assert turboquant.is_active(model), "Expected TurboQuant to be active"

        import torch
        inputs = tokenizer("What is 2 + 2? Answer with just the number.", return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        output_ids = model.generate(inputs["input_ids"], max_new_tokens=32)
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        turboquant.print_telemetry(model)

        telemetry = turboquant.last_telemetry(model)
        metrics = turboquant.last_metrics(model)

        turboquant.deactivate(model)
        assert not turboquant.is_active(model), "Expected TurboQuant to be deactivated"

        return {
            "model": model_id,
            "bits": bits,
            "text": text,
            "telemetry": telemetry,
            "has_metrics": metrics is not None,
            "activate_deactivate_ok": True,
        }


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    variant: str = "qmse_packed",
    bits: int = 3,
    prompt: str = "Explain KV cache compression in one short paragraph.",
    max_new_tokens: int = 128,
    num_outlier_channels: int = 0,
    outlier_extra_bits: int = 1,
    use_qjl_keys: bool = False,
    quantize_decode: bool = False,
    norm_guard: bool = True,
    profile_channels: bool = False,
    memory_benchmark: bool = False,
    output_test: bool = False,
    activate_test: bool = False,
    prompt_tokens: int = 2048,
) -> None:
    import json

    if activate_test:
        result = SmokeRunner().activate_test.remote(model_id=model, bits=bits)
    elif output_test:
        result = SmokeRunner().output_test.remote(
            model=model,
            bits=bits,
            max_new_tokens=64,
            use_qjl_keys=use_qjl_keys,
            quantize_decode=quantize_decode,
        )
    elif memory_benchmark:
        result = SmokeRunner().memory_benchmark.remote(
            model=model,
            bits=bits,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            use_qjl_keys=use_qjl_keys,
            quantize_decode=quantize_decode,
        )
    elif profile_channels:
        result = SmokeRunner().profile_channels.remote(model=model, prompt=prompt)
    else:
        result = SmokeRunner().run.remote(
            model=model,
            variant=variant,
            bits=bits,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            num_outlier_channels=num_outlier_channels,
            outlier_extra_bits=outlier_extra_bits,
            use_qjl_keys=use_qjl_keys,
            quantize_decode=quantize_decode,
            norm_guard=norm_guard,
        )
    print(json.dumps(result, indent=2, default=str))
