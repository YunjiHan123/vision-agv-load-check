from __future__ import annotations

from pathlib import Path


def normalize_input_path(path: str | Path) -> Path:
    """Normalize user-provided file system paths."""
    return Path(path).expanduser()
