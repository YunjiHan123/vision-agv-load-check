from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from src.utils.io import load_image


_SIDE_CAR_SUFFIXES = (
    ".perspective.yaml",
    ".perspective.yml",
    ".perspective.json",
)


def _default_output_dir() -> Path:
    return Path("data/processed/perspective")


def _order_points(points: np.ndarray) -> np.ndarray:
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)

    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def _load_sidecar_config(input_path: Path) -> dict[str, Any] | None:
    for suffix in _SIDE_CAR_SUFFIXES:
        candidate = input_path.with_name(f"{input_path.stem}{suffix}")
        if not candidate.exists():
            continue

        text = candidate.read_text(encoding="utf-8")
        if candidate.suffix == ".json":
            data = json.loads(text)
        else:
            data = yaml.safe_load(text)

        if not isinstance(data, dict):
            raise ValueError(f"Perspective config must be a mapping: {candidate}")

        data["config_path"] = str(candidate)
        return data

    return None


def _coerce_points(raw_points: Any) -> np.ndarray:
    points = np.asarray(raw_points, dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError("Perspective transform requires exactly 4 (x, y) points.")
    return _order_points(points)


def _compute_destination_points(source_points: np.ndarray, output_size: tuple[int, int] | None) -> np.ndarray:
    if output_size is None:
        width_top = np.linalg.norm(source_points[1] - source_points[0])
        width_bottom = np.linalg.norm(source_points[2] - source_points[3])
        height_right = np.linalg.norm(source_points[2] - source_points[1])
        height_left = np.linalg.norm(source_points[3] - source_points[0])
        width = max(int(round(max(width_top, width_bottom))), 1)
        height = max(int(round(max(height_left, height_right))), 1)
    else:
        width = max(int(output_size[0]), 1)
        height = max(int(output_size[1]), 1)

    return np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )


def _build_visualization(
    original_image: np.ndarray,
    warped_image: np.ndarray,
    source_points: np.ndarray,
) -> np.ndarray:
    original_panel = original_image.copy()
    cv2.polylines(original_panel, [source_points.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=3)

    target_height = original_panel.shape[0]
    warped_width = max(int(round(warped_image.shape[1] * (target_height / warped_image.shape[0]))), 1)
    resized_warped = cv2.resize(warped_image, (warped_width, target_height), interpolation=cv2.INTER_LINEAR)

    return np.concatenate([original_panel, resized_warped], axis=1)


def apply_perspective_transform(input_path: Path, output_dir: Path | None = None) -> dict[str, Any]:
    """Warp the image using manually defined quadrilateral points when available."""
    input_path = Path(input_path)
    result: dict[str, Any] = {
        "stage": "perspective_transform",
        "input_path": str(input_path),
        "status": "not_applied",
        "applied": False,
        "image": None,
        "original_image": None,
        "transform_matrix": None,
        "source_points": [],
        "destination_points": [],
        "warped_path": None,
        "visualization_path": None,
    }

    if input_path.is_dir():
        result["status"] = "skipped_directory"
        return result

    image = load_image(input_path)
    if image is None:
        result["status"] = "image_load_failed"
        return result

    result["original_image"] = image
    result["image"] = image
    result["original_shape"] = list(image.shape)

    config = _load_sidecar_config(input_path)
    if config is None:
        result["status"] = "points_not_defined"
        return result

    if "source_points" not in config:
        raise ValueError("Perspective config must include 'source_points'.")

    source_points = _coerce_points(config["source_points"])
    raw_output_size = config.get("output_size")
    output_size = tuple(raw_output_size) if raw_output_size is not None else None
    destination_points = _compute_destination_points(source_points, output_size)
    transform_matrix = cv2.getPerspectiveTransform(source_points, destination_points)
    width = int(destination_points[1][0] + 1)
    height = int(destination_points[2][1] + 1)
    warped_image = cv2.warpPerspective(image, transform_matrix, (width, height))

    output_base_dir = output_dir if output_dir is not None else _default_output_dir()
    output_base_dir.mkdir(parents=True, exist_ok=True)
    warped_path = output_base_dir / f"{input_path.stem}_warped.png"
    visualization_path = output_base_dir / f"{input_path.stem}_perspective_debug.png"

    cv2.imwrite(str(warped_path), warped_image)
    visualization = _build_visualization(image, warped_image, source_points)
    cv2.imwrite(str(visualization_path), visualization)

    result.update(
        {
            "status": "applied",
            "applied": True,
            "config_path": config.get("config_path"),
            "image": warped_image,
            "warped_shape": list(warped_image.shape),
            "transform_matrix": transform_matrix.tolist(),
            "source_points": source_points.tolist(),
            "destination_points": destination_points.tolist(),
            "warped_path": str(warped_path),
            "visualization_path": str(visualization_path),
        }
    )
    return result
