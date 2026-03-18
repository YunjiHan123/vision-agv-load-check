from __future__ import annotations

from pathlib import Path


STATUS_ALIASES = {
    "normal": "normal",
    "shifted": "misaligned",
    "tilted": "misaligned",
    "misaligned": "misaligned",
    "fallen": "fallen",
    "abnormal": "misaligned",
}


def normalize_status_label(status: str) -> str:
    normalized = status.strip().lower()
    return STATUS_ALIASES.get(normalized, normalized or "unknown")


def infer_status_from_path(input_path: str | Path) -> str:
    stem = Path(input_path).stem.lower()
    for label in ("fallen", "shifted", "tilted", "misaligned", "normal"):
        if label in stem:
            return normalize_status_label(label)
    return "unknown"
