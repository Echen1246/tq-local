from __future__ import annotations

import argparse
import json

from turboquant.benchmarks.paper import as_serializable_benchmarks
from turboquant.sources import as_serializable_sources


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="turboquant")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("sources", help="Print official upstream sources used by the project.")
    subparsers.add_parser("benchmarks", help="Print the paper-faithful benchmark manifest.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "sources":
        print(json.dumps(as_serializable_sources(), indent=2, sort_keys=True))
        return

    if args.command == "benchmarks":
        print(json.dumps(as_serializable_benchmarks(), indent=2, sort_keys=True))
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
