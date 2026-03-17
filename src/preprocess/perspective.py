from __future__ import annotations

from pathlib import Path


def apply_perspective_transform(input_path: Path) -> dict[str, str]:
    """Placeholder for perspective correction before detection."""
    return {
        "stage": "perspective_transform",
        "input_path": str(input_path),
        "status": "not_implemented",
    }
