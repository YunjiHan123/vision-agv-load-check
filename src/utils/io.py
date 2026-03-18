from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def normalize_input_path(path: str | Path) -> Path:
    """Normalize user-provided file system paths."""
    return Path(path).expanduser()


def load_image(path: str | Path) -> np.ndarray | None:
    """Load an image with EXIF orientation applied and return a BGR ndarray."""
    resolved = normalize_input_path(path)
    if not resolved.is_file():
        return None

    with Image.open(resolved) as image:
        corrected = ImageOps.exif_transpose(image).convert("RGB")
        rgb = np.array(corrected)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
