from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def log_experiment_event(log_root: Path, event: str, payload: dict[str, Any]) -> Path:
    record = {
        "timestamp_utc": utc_now_iso(),
        "event": event,
        **payload,
    }
    path = log_root / "experiment_log.jsonl"
    append_jsonl(path, record)
    return path
