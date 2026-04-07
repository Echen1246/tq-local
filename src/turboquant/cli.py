from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

from turboquant import (
    TurboQuantSession,
    inspect_transformers_model_compatibility,
    load_transformers_model,
)
from turboquant.adapters.transformers import TransformersLoadConfig
from turboquant.constants import DEFAULT_MAX_NEW_TOKENS

_DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
_DEFAULT_BITS = 4
_SUPPORTED_BITS = [2, 3, 4]

_TESTED_MODELS = [
    {
        "id": "meta-llama/Llama-3.1-8B-Instruct",
        "status": "fully working",
        "notes": "74% savings, all layers quantize cleanly, Q_prod default",
    },
    {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "status": "works with norm guard",
        "notes": "3/28 layers kept dense due to extreme key norms, 71.5% savings",
    },
]

def _format_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "n/a"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"

def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))

def _read_prompt(args) -> str:
    if args.prompt is not None:
        return args.prompt
    if getattr(args, "prompt_file", None) is not None:
        return Path(args.prompt_file).read_text()
    raise ValueError("Either --prompt or --prompt-file is required.")

def _common_load_kwargs(args) -> dict[str, Any]:
    return {
        "revision": getattr(args, "revision", None),
        "dtype": getattr(args, "dtype", "auto"),
        "device_map": getattr(args, "device_map", "auto"),
        "attn_implementation": getattr(args, "attn_implementation", "sdpa"),
        "trust_remote_code": getattr(args, "trust_remote_code", False),
        "token": getattr(args, "token", None),
        "cache_dir": getattr(args, "cache_dir", None),
    }

def _colors():
    if not sys.stdout.isatty():
        return {"C": "", "B": "", "D": "", "R": "", "G": "", "Y": ""}
    return {
        "C": "\033[36m", "B": "\033[1m", "D": "\033[2m",
        "R": "\033[0m", "G": "\033[32m", "Y": "\033[33m",
    }

def _gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"cuda_available": False, "devices": []}
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                mem = getattr(props, "total_memory", None)
                if mem is None:
                    mem = getattr(props, "total_mem", 0)
                info["devices"].append({
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(mem / (1024**3), 1),
                    "compute_capability": f"{props.major}.{props.minor}",
                })
    except ImportError:
        info["torch_version"] = None
    return info

def _model_recommendations(vram_gb: float) -> list[str]:
    recs = []
    if vram_gb >= 80:
        recs.append("Llama-3.1-70B at 3-bit (full context)")
    if vram_gb >= 48:
        recs.append("Llama-3.1-8B at 3-bit (100K+ context)")
    if vram_gb >= 24:
        recs.append("Llama-3.1-8B at 4-bit (up to ~73K context)")
    if vram_gb >= 16:
        recs.append("Llama-3.1-8B at 3-bit (up to ~32K context)")
    if not recs:
        recs.append("Consider using Modal/Colab for cloud GPU access")
    return recs

def _welcome() -> int:
    from turboquant import __version__

    c = _colors()
    gpu = _gpu_info()
    w = min(shutil.get_terminal_size((80, 24)).columns, 64)
    bar = "─" * w

    print(f"""{c['C']}
  ╔╦╗┬ ┬┬─┐┌┐ ┌─┐╔═╗ ┬ ┬┌─┐┌┐┌┌┬┐
   ║ │ │├┬┘├┴┐│ │║═╬╗│ │├─┤│││ │
   ╩ └─┘┴└─└─┘└─┘╚═╝╚└─┘┴ ┴┘└┘ ┴{c['R']}
""")
    print(f"  {c['B']}v{__version__}{c['R']}  —  KV cache compression for HuggingFace Transformers")
    print(f"  {c['D']}Paper-accurate Q_prod with fused Triton kernel{c['R']}")
    print(f"  {c['D']}github.com/Echen1246/local-turboquant{c['R']}")
    print()
    print(f"  {bar}")
    print()

    print(f"  {c['B']}Quick start{c['R']}")
    print()
    print(f"  {c['G']}1.{c['R']}  One-shot prompt:")
    print(f"      {c['D']}turboquant run --prompt \"Explain quantum computing\"{c['R']}")
    print()
    print(f"  {c['G']}2.{c['R']}  Interactive session:")
    print(f"      {c['D']}turboquant attach{c['R']}")
    print()
    print(f"  {c['G']}3.{c['R']}  Python API:")
    print(f"      {c['D']}from turboquant import TurboQuantSession{c['R']}")
    print(f"      {c['D']}s = TurboQuantSession.from_pretrained(\"meta-llama/Llama-3.1-8B-Instruct\"){c['R']}")
    print(f"      {c['D']}print(s.generate(\"Hello\"))  # KV cache auto-compressed{c['R']}")
    print()
    print(f"  {bar}")
    print()

    print(f"  {c['B']}Defaults{c['R']}")
    print(f"    Model:    {_DEFAULT_MODEL}")
    print(f"    Bits:     {_DEFAULT_BITS}-bit Q_prod (3-bit MSE keys + 1-bit QJL, 4-bit MSE values)")
    print(f"    Savings:  ~74% KV cache VRAM reduction")
    print(f"    Kernel:   Fused Triton (GPU) / chunked PyTorch (CPU fallback)")
    print()

    print(f"  {bar}")
    print()
    print(f"  {c['B']}Commands{c['R']}")
    print()
    print(f"  {c['Y']}turboquant run{c['R']}         Run a single prompt (default model + Q_prod)")
    print(f"  {c['Y']}turboquant attach{c['R']}      Load model and prompt interactively")
    print(f"  {c['Y']}turboquant setup{c['R']}       GPU detection, system info, recommendations")
    print(f"  {c['Y']}turboquant telemetry{c['R']}   Display saved run telemetry")
    print()

    print(f"  {bar}")
    print()
    print(f"  {c['B']}System{c['R']}")

    if gpu["cuda_available"] and gpu["devices"]:
        dev = gpu["devices"][0]
        print(f"  {c['G']}✓{c['R']} GPU: {dev['name']} ({dev['total_memory_gb']} GB)")
        recs = _model_recommendations(dev["total_memory_gb"])
        for r in recs[:2]:
            print(f"    {c['D']}{r}{c['R']}")
    elif platform.system() == "Darwin":
        print(f"  {c['Y']}!{c['R']} No CUDA GPU (macOS) — use Colab/Modal for inference")
    else:
        print(f"  {c['Y']}!{c['R']} No CUDA GPU — TurboQuant requires NVIDIA GPU")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        print(f"  {c['G']}✓{c['R']} HF_TOKEN set")
    else:
        print(f"  {c['D']}i{c['R']} HF_TOKEN not set {c['D']}(needed for gated models like Llama){c['R']}")

    print()
    return 0

def _handle_setup(args) -> int:
    from turboquant import __version__

    gpu = _gpu_info()

    if args.json:
        _print_json({
            "version": __version__,
            "python": sys.version,
            "platform": platform.platform(),
            "gpu": gpu,
            "defaults": {
                "model": _DEFAULT_MODEL,
                "bits": _DEFAULT_BITS,
                "qjl_keys": True,
                "quantize_decode": True,
            },
            "tested_models": _TESTED_MODELS,
        })
        return 0

    c = _colors()
    w = min(shutil.get_terminal_size((80, 24)).columns, 60)
    bar = "─" * w

    print()
    print(f"  {c['B']}TurboQuant v{__version__}{c['R']}")
    print(bar)

    print()
    print(f"  {c['B']}System{c['R']}")
    print(f"    Python:       {sys.version.split()[0]}")
    print(f"    Platform:     {platform.platform()}")

    if gpu.get("torch_version"):
        print(f"    PyTorch:      {gpu['torch_version']}")
    try:
        import transformers
        print(f"    Transformers: {transformers.__version__}")
    except ImportError:
        print(f"    Transformers: {c['Y']}not installed{c['R']}")

    print()
    print(f"  {c['B']}GPU{c['R']}")
    if gpu["cuda_available"] and gpu["devices"]:
        print(f"    CUDA:         {gpu.get('cuda_version', 'unknown')}")
        for dev in gpu["devices"]:
            print(f"    Device {dev['index']}:     {dev['name']} "
                  f"({dev['total_memory_gb']} GB, compute {dev['compute_capability']})")
    elif platform.system() == "Darwin":
        print(f"    {c['Y']}No CUDA (macOS) — use Colab or Modal{c['R']}")
    else:
        print(f"    {c['Y']}No CUDA GPU detected{c['R']}")

    try:
        from turboquant.runtime.triton_kernels import triton_available
        if triton_available():
            print(f"    Triton:       {c['G']}available{c['R']} (fused Q_prod kernel)")
        else:
            print(f"    Triton:       not available (PyTorch fallback)")
    except ImportError:
        print(f"    Triton:       not available (PyTorch fallback)")

    print()
    print(f"  {c['B']}Defaults{c['R']}")
    print(f"    Model:        {_DEFAULT_MODEL}")
    print(f"    Quantization: {_DEFAULT_BITS}-bit Q_prod (paper-accurate)")
    print(f"    Bit widths:   4 (near-lossless), 3 (very good), 2 (experimental)")

    if gpu["cuda_available"] and gpu["devices"]:
        max_vram = max(d["total_memory_gb"] for d in gpu["devices"])
        recs = _model_recommendations(max_vram)
        print()
        print(f"  {c['B']}Recommended for {max_vram} GB VRAM{c['R']}")
        for r in recs:
            print(f"    - {r}")

    print()
    print(f"  {c['B']}Tested models{c['R']}")
    for m in _TESTED_MODELS:
        icon = f"{c['G']}✓{c['R']}" if m["status"] == "fully working" else f"{c['Y']}~{c['R']}"
        print(f"    {icon} {m['id']}")
        print(f"      {c['D']}{m['notes']}{c['R']}")

    print()
    return 0

def _handle_telemetry(args) -> int:
    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        return 1
    data = json.loads(path.read_text())

    telemetry = data.get("telemetry")
    metrics = data.get("metrics")
    if telemetry is None and metrics is None:
        print("No telemetry found. Run with `turboquant run --json` to produce one.")
        return 1

    model = data.get("model", "unknown")
    bits = data.get("bits")
    print(f"Model:   {model}")
    if bits:
        print(f"Bits:    {bits}")
    print()

    if telemetry:
        dense = telemetry.get("dense_kv_bytes")
        packed = telemetry.get("packed_actual_bytes") or telemetry.get("packed_estimate_bytes")
        savings = telemetry.get("payload_savings_percent")

        print("Cache:")
        print(f"  Dense KV:  {_format_bytes(dense)}")
        print(f"  Packed KV: {_format_bytes(packed)}")
        if savings is not None:
            print(f"  Savings:   {savings:.1f}%")
        print()

        post_setup = telemetry.get("post_cache_setup_allocated_bytes")
        peak = telemetry.get("peak_allocated_bytes")
        print("VRAM:")
        print(f"  After setup: {_format_bytes(post_setup)}")
        print(f"  Peak:        {_format_bytes(peak)}")
        print()

        gen_s = telemetry.get("generation_seconds")
        comp_tok = telemetry.get("completion_tokens")
        prompt_tok = telemetry.get("prompt_tokens")
        print("Timing:")
        if prompt_tok:
            print(f"  Prompt:      {prompt_tok} tokens")
        if comp_tok:
            print(f"  Generated:   {comp_tok} tokens")
        if gen_s:
            print(f"  Gen time:    {gen_s:.3f}s")
            if comp_tok and gen_s > 0:
                print(f"  Speed:       {comp_tok / gen_s:.1f} tok/s")

    if metrics:
        recon = metrics.get("reconstruction_quality")
        if recon:
            print()
            print("Quality:")
            for key in ("avg_key_cosine_sim", "avg_val_cosine_sim"):
                v = recon.get(key)
                if v is not None:
                    label = key.replace("avg_", "").replace("_", " ").title()
                    print(f"  {label}: {v:.6f}")

    return 0

def _print_telemetry_summary(telemetry: dict) -> None:
    c = _colors()
    dense = telemetry.get("dense_kv_bytes")
    packed = telemetry.get("packed_actual_bytes") or telemetry.get("packed_estimate_bytes")
    savings = telemetry.get("payload_savings_percent")
    gen_s = telemetry.get("generation_seconds")
    comp_tok = telemetry.get("completion_tokens")

    parts = []
    if savings is not None:
        parts.append(f"{savings:.0f}% KV saved")
    if dense and packed:
        saved_mb = (dense - packed) / (1024 * 1024)
        parts.append(f"{saved_mb:.0f} MB freed")
    if gen_s and comp_tok:
        parts.append(f"{comp_tok / gen_s:.1f} tok/s")
    elif gen_s:
        parts.append(f"{gen_s:.1f}s")

    if parts:
        print(f"\n{c['D']}[TurboQuant] {' | '.join(parts)}{c['R']}")

def _handle_attach(args) -> int:
    c = _colors()
    model_id = args.model

    print()
    print(f"  {c['B']}Loading {model_id}{c['R']}")

    qjl = not args.no_qjl
    quant_decode = not args.no_quantize_decode
    bits = args.bits

    mode = "Q_prod" if qjl else "Q_mse"
    print(f"  {c['D']}{bits}-bit {mode} | quantize_decode={'on' if quant_decode else 'off'}{c['R']}")
    print()

    session = TurboQuantSession.from_pretrained(
        model_id,
        variant="qmse_packed",
        bits=bits,
        use_qjl_keys=qjl,
        quantize_decode=quant_decode,
        norm_guard=not getattr(args, "no_norm_guard", False),
        **_common_load_kwargs(args),
    )

    gpu = _gpu_info()
    if gpu["cuda_available"] and gpu["devices"]:
        dev = gpu["devices"][0]
        print(f"  {c['G']}✓{c['R']} Model loaded on {dev['name']}")
    else:
        print(f"  {c['G']}✓{c['R']} Model loaded")

    print(f"  {c['G']}✓{c['R']} TurboQuant {bits}-bit {mode} active")
    print(f"  {c['D']}Type a prompt and press Enter. Ctrl+C or 'exit' to quit.{c['R']}")
    print()

    max_tokens = args.max_new_tokens

    while True:
        try:
            prompt = input(f"{c['G']}> {c['R']}")
        except (KeyboardInterrupt, EOFError):
            print()
            break

        prompt = prompt.strip()
        if not prompt or prompt.lower() in ("exit", "quit", ":q"):
            break

        if prompt.startswith("/tokens "):
            try:
                max_tokens = int(prompt.split()[1])
                print(f"{c['D']}Max tokens set to {max_tokens}{c['R']}")
            except (ValueError, IndexError):
                print(f"{c['Y']}Usage: /tokens <number>{c['R']}")
            continue

        if prompt == "/stats":
            telemetry = session.last_telemetry()
            if telemetry:
                _print_telemetry_summary(telemetry)
            else:
                print(f"{c['D']}No telemetry yet — send a prompt first.{c['R']}")
            continue

        if prompt == "/help":
            print(f"  {c['D']}/tokens N   — set max generation tokens (current: {max_tokens}){c['R']}")
            print(f"  {c['D']}/stats     — show last generation telemetry{c['R']}")
            print(f"  {c['D']}/help      — show this help{c['R']}")
            print(f"  {c['D']}exit       — quit{c['R']}")
            continue

        text = session.generate(prompt=prompt, max_new_tokens=max_tokens)
        print(text)

        telemetry = session.last_telemetry()
        if telemetry:
            _print_telemetry_summary(telemetry)
        print()

    return 0

def _handle_run(args) -> int:
    prompt = _read_prompt(args)
    qjl = not args.no_qjl
    quant_decode = not args.no_quantize_decode

    session = TurboQuantSession.from_pretrained(
        args.model,
        variant="qmse_packed" if not args.baseline else "baseline",
        bits=args.bits,
        rotation_seed=getattr(args, "rotation_seed", 0),
        use_qjl_keys=qjl,
        quantize_decode=quant_decode,
        norm_guard=not getattr(args, "no_norm_guard", False),
        **_common_load_kwargs(args),
    )
    text = session.generate(prompt=prompt, max_new_tokens=args.max_new_tokens)
    telemetry = session.last_telemetry()
    metrics = session.last_metrics()

    if args.json:
        _print_json({
            "model": args.model,
            "bits": args.bits,
            "qjl": qjl,
            "text": text,
            "metrics": metrics,
            "telemetry": telemetry,
        })
        return 0

    print(text)
    if telemetry:
        _print_telemetry_summary(telemetry)
    if args.verbose and metrics:
        print("\nMetrics:")
        _print_json(metrics)

    return 0

def _add_load_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--revision", default=None, help=argparse.SUPPRESS)
    p.add_argument("--dtype", default="auto", help="Model dtype (default: auto)")
    p.add_argument("--device-map", default="auto", help=argparse.SUPPRESS)
    p.add_argument("--attn-implementation", default="sdpa", help=argparse.SUPPRESS)
    p.add_argument("--trust-remote-code", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--token", default=None, help="HuggingFace token for gated models")
    p.add_argument("--cache-dir", default=None, help=argparse.SUPPRESS)

def _add_quant_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--model", default=_DEFAULT_MODEL,
        help=f"Model ID or path (default: {_DEFAULT_MODEL})",
    )
    p.add_argument("--bits", type=int, default=_DEFAULT_BITS, help=f"Bit width (default: {_DEFAULT_BITS})")
    p.add_argument("--no-qjl", action="store_true", help="Disable QJL (use Q_mse instead of Q_prod)")
    p.add_argument(
        "--no-quantize-decode", action="store_true",
        help="Keep decode tokens dense (don't re-quantize)",
    )
    p.add_argument("--no-norm-guard", action="store_true", help=argparse.SUPPRESS)
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Max tokens to generate (default: {DEFAULT_MAX_NEW_TOKENS})",
    )

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="turboquant",
        description="TurboQuant — KV cache compression for HuggingFace Transformers.",
    )
    sub = parser.add_subparsers(dest="command", required=False)

    # setup
    sp_setup = sub.add_parser("setup", help="GPU detection, system info, recommendations")
    sp_setup.add_argument("--json", action="store_true")
    sp_setup.set_defaults(func=_handle_setup)

    # run
    sp_run = sub.add_parser("run", help="Run a single prompt with TurboQuant compression")
    _add_load_args(sp_run)
    _add_quant_args(sp_run)
    prompt_group = sp_run.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", default=None, help="Prompt text")
    prompt_group.add_argument("--prompt-file", default=None, help="Read prompt from file")
    sp_run.add_argument("--baseline", action="store_true", help="Run without compression (for comparison)")
    sp_run.add_argument("--json", action="store_true", help="Output as JSON")
    sp_run.add_argument("-v", "--verbose", action="store_true", help="Show full metrics")
    sp_run.add_argument("--rotation-seed", type=int, default=0, help=argparse.SUPPRESS)
    sp_run.set_defaults(func=_handle_run)

    # attach
    sp_attach = sub.add_parser("attach", help="Load model and prompt interactively")
    _add_load_args(sp_attach)
    _add_quant_args(sp_attach)
    sp_attach.set_defaults(func=_handle_attach)

    # telemetry
    sp_telem = sub.add_parser("telemetry", help="Display saved telemetry from a JSON run")
    sp_telem.add_argument("file", help="Path to JSON from `turboquant run --json`")
    sp_telem.set_defaults(func=_handle_telemetry)

    return parser

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        return _welcome()
    return int(args.func(args))

if __name__ == "__main__":
    raise SystemExit(main())
