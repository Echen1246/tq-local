from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-").lower()
    return slug or "run"


def resolve_run_name(prefix: str, run_name: str | None = None) -> str:
    if run_name:
        return slugify(run_name)
    return f"{prefix}-{utc_timestamp()}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
