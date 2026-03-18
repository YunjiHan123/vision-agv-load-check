from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

try:
    from ultralytics import YOLO
except ModuleNotFoundError:
    YOLO = None


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Detector config must be a mapping: {path}")
    return data


def _resolve_detector_config_path(config_path: Path) -> Path:
    config_path = Path(config_path)
    if config_path.name == "yolo.yaml":
        return config_path

    fallback = config_path.parent / "yolo.yaml"
    return fallback if fallback.exists() else Path("configs/yolo.yaml")


def _load_raw_image(image_context: dict[str, Any]) -> np.ndarray | None:
    raw_image = image_context.get("raw_image")
    if isinstance(raw_image, np.ndarray):
        return raw_image

    input_path = image_context.get("raw_input_path")
    if isinstance(input_path, str):
        path = Path(input_path)
        if path.is_file():
            return cv2.imread(str(path))

    return None


def _build_brown_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([5, 20, 20]), np.array([35, 255, 220]))
    kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)


def _divider_count(gray_roi: np.ndarray) -> int:
    edges = cv2.Canny(gray_roi, 80, 180)
    column_strength = (edges > 0).sum(axis=0)
    threshold = max(int(gray_roi.shape[0] * 0.05), 10)
    indices = np.where(column_strength > threshold)[0]
    if len(indices) == 0:
        return 0

    group_count = 1
    previous = int(indices[0])
    for index in indices[1:]:
        current = int(index)
        if current - previous > 10:
            group_count += 1
        previous = current
    return group_count


def _estimate_count_from_dividers(divider_count: int) -> int:
    if divider_count <= 0:
        return 1
    if divider_count in {3, 4}:
        return 4
    if divider_count <= 2:
        return divider_count + 1
    return (divider_count // 2) + 1


def _infer_status_from_geometry(x: int, w: int, h: int, solidity: float) -> str:
    aspect_ratio = w / max(h, 1)
    if aspect_ratio > 1.0:
        return "fallen"
    if x <= 5 or solidity > 0.75:
        return "misaligned"
    return "normal"


def _build_fallback_detections(image_context: dict[str, Any]) -> list[dict[str, object]]:
    raw_image = _load_raw_image(image_context)
    if raw_image is None:
        return []

    resized = cv2.resize(raw_image, (768, 1024), interpolation=cv2.INTER_AREA)
    mask = _build_brown_mask(resized)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) > 50000]
    if not contours:
        return []

    stack_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(stack_contour)
    area = cv2.contourArea(stack_contour)
    hull_area = cv2.contourArea(cv2.convexHull(stack_contour))
    solidity = float(area / hull_area) if hull_area else 0.0
    gray_roi = cv2.cvtColor(resized[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)
    divider_count = _divider_count(gray_roi)
    estimated_count = _estimate_count_from_dividers(divider_count)
    status_hint = _infer_status_from_geometry(x, w, h, solidity)

    detections: list[dict[str, object]] = []
    segment_width = max(w // estimated_count, 1)
    for index in range(estimated_count):
        left = x + (index * segment_width)
        right = x + w if index == estimated_count - 1 else min(x + ((index + 1) * segment_width), x + w)
        detections.append(
            {
                "class_name": "mini_box",
                "confidence": 0.35,
                "bbox": [int(left), int(y), int(right), int(y + h)],
                "scene_status_hint": status_hint,
                "detector_source": "heuristic_fallback",
                "stack_bbox": [int(x), int(y), int(x + w), int(y + h)],
                "stack_solidity": round(solidity, 4),
                "divider_count": divider_count,
            }
        )

    return detections


def _run_yolo_detection(
    image_context: dict[str, Any],
    detector_config: dict[str, Any],
    detector_config_path: Path,
) -> list[dict[str, object]]:
    raw_image = _load_raw_image(image_context)
    if raw_image is None or YOLO is None:
        return []

    model_config = detector_config.get("model", {})
    inference_config = detector_config.get("inference", {})
    weights_value = model_config.get("weights")
    if not isinstance(weights_value, str):
        return []

    weights_path = Path(weights_value)
    if not weights_path.is_absolute():
        weights_path = (detector_config_path.parent.parent / weights_path).resolve()
    if not weights_path.exists():
        return []

    model = YOLO(str(weights_path))
    results = model.predict(
        source=raw_image,
        conf=float(inference_config.get("confidence_threshold", 0.25)),
        iou=float(inference_config.get("iou_threshold", 0.45)),
        imgsz=int(inference_config.get("image_size", 1280)),
        verbose=False,
    )
    if not results:
        return []

    boxes = getattr(results[0], "boxes", None)
    if boxes is None:
        return []

    detections: list[dict[str, object]] = []
    class_name = str(model_config.get("class_name", "mini_box"))
    for box in boxes:
        xyxy = box.xyxy[0].tolist()
        confidence = float(box.conf[0].item()) if box.conf is not None else 0.0
        detections.append(
            {
                "class_name": class_name,
                "confidence": confidence,
                "bbox": [int(value) for value in xyxy],
                "detector_source": "ultralytics",
            }
        )
    return detections


def detect_lot_labels(image: dict[str, Any], config_path: Path) -> list[dict[str, object]]:
    """Run YOLO when weights are available, otherwise use a contour-based fallback detector."""
    detector_config_path = _resolve_detector_config_path(config_path)
    detector_config = _load_yaml(detector_config_path) if detector_config_path.exists() else {}

    yolo_detections = _run_yolo_detection(image, detector_config, detector_config_path)
    if yolo_detections:
        return yolo_detections

    return _build_fallback_detections(image)
