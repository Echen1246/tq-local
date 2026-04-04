from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from turboquant import (
    TurboQuantSession,
    inspect_transformers_model_compatibility,
    load_transformers_model,
)
from turboquant.adapters.transformers import TransformersLoadConfig


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
    if args.prompt_file is not None:
        return Path(args.prompt_file).read_text()
    raise ValueError("Either --prompt or --prompt-file is required.")


def _common_load_kwargs(args) -> dict[str, Any]:
    return {
        "revision": args.revision,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "attn_implementation": args.attn_implementation,
        "trust_remote_code": args.trust_remote_code,
        "token": args.token,
        "cache_dir": args.cache_dir,
    }


def _handle_inspect(args) -> int:
    tokenizer, model = load_transformers_model(
        TransformersLoadConfig(
            model_id_or_path=args.model,
            **_common_load_kwargs(args),
        )
    )
    _ = tokenizer
    report = inspect_transformers_model_compatibility(model).to_dict()
    if args.json:
        _print_json(report)
        return 0

    print(f"model: {args.model}")
    print(f"backend: {report['backend']}")
    print(f"compatible: {report['compatible']}")
    if report["reasons"]:
        print("reasons:")
        for item in report["reasons"]:
            print(f"- {item}")
    if report["warnings"]:
        print("warnings:")
        for item in report["warnings"]:
            print(f"- {item}")
    if report["details"]:
        print("details:")
        for key, value in report["details"].items():
            print(f"- {key}: {value}")
    return 0


def _handle_run(args) -> int:
    prompt = _read_prompt(args)
    session = TurboQuantSession.from_pretrained(
        args.model,
        variant=args.variant,
        bits=args.bits,
        rotation_seed=args.rotation_seed,
        **_common_load_kwargs(args),
    )
    text = session.generate(
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
    )
    metrics = session.last_metrics()
    telemetry = session.last_telemetry()
    payload = {
        "model": args.model,
        "variant": args.variant,
        "bits": args.bits if args.variant != "baseline" else None,
        "text": text,
        "metrics": metrics,
        "telemetry": telemetry,
    }
    if args.json:
        _print_json(payload)
        return 0

    print(text)
    if args.show_metrics and metrics is not None:
        print("\nmetrics:")
        _print_json(metrics)
    if args.show_telemetry and telemetry is not None:
        print("\ntelemetry:")
        print(f"- dense_kv_bytes: {_format_bytes(telemetry['dense_kv_bytes'])}")
        print(f"- packed_estimate_bytes: {_format_bytes(telemetry['packed_estimate_bytes'])}")
        print(f"- packed_actual_bytes: {_format_bytes(telemetry['packed_actual_bytes'])}")
        print(f"- payload_savings_percent: {telemetry['payload_savings_percent']}")
        print(
            f"- post_cache_setup_allocated_bytes: "
            f"{_format_bytes(telemetry['post_cache_setup_allocated_bytes'])}"
        )
        print(f"- peak_allocated_bytes: {_format_bytes(telemetry['peak_allocated_bytes'])}")
        print(f"- generation_seconds: {telemetry['generation_seconds']}")
        print(f"- quantization_seconds: {telemetry['quantization_seconds']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="turboquant",
        description="TurboQuant-style KV-cache compression for Hugging Face Transformers.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_load_args(target) -> None:
        target.add_argument("--model", required=True, help="Hugging Face model ID or local model path.")
        target.add_argument("--revision", default=None, help="Optional model revision.")
        target.add_argument("--dtype", default="auto", help="Transformers dtype argument. Default: auto.")
        target.add_argument("--device-map", default="auto", help="Transformers device_map argument.")
        target.add_argument(
            "--attn-implementation",
            default="sdpa",
            help="Attention backend to request from Transformers. Default: sdpa.",
        )
        target.add_argument(
            "--trust-remote-code",
            action="store_true",
            help="Pass trust_remote_code=True when loading the model/tokenizer.",
        )
        target.add_argument("--token", default=None, help="Optional Hugging Face token.")
        target.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory.")

    inspect_parser = subparsers.add_parser("inspect", help="Load a model and report TurboQuant compatibility.")
    add_load_args(inspect_parser)
    inspect_parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    inspect_parser.set_defaults(func=_handle_inspect)

    run_parser = subparsers.add_parser("run", help="Run a prompt through a Transformers model with TurboQuant.")
    add_load_args(run_parser)
    prompt_group = run_parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", default=None, help="Prompt text.")
    prompt_group.add_argument("--prompt-file", default=None, help="Read the prompt from a file.")
    run_parser.add_argument(
        "--variant",
        default="qmse_packed",
        choices=["baseline", "qmse", "qmse_packed"],
        help="Generation variant to use. Default: qmse_packed.",
    )
    run_parser.add_argument("--bits", type=int, default=3, help="Quantization bits for qmse variants.")
    run_parser.add_argument("--rotation-seed", type=int, default=0, help="Rotation seed. Default: 0.")
    run_parser.add_argument("--max-new-tokens", type=int, default=256, help="Max generated tokens.")
    run_parser.add_argument("--json", action="store_true", help="Print full output as JSON.")
    run_parser.add_argument(
        "--show-metrics",
        action="store_true",
        help="Print the raw generation metrics after the response.",
    )
    run_parser.add_argument(
        "--show-telemetry",
        action="store_true",
        help="Print a concise cache/memory telemetry summary after the response.",
    )
    run_parser.set_defaults(func=_handle_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
