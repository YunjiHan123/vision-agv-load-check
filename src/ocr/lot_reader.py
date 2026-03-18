from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

try:
    from paddleocr import PaddleOCR
except ModuleNotFoundError:
    PaddleOCR = None


_OCR_INSTANCE: Any | None = None
_LOT_PATTERN = re.compile(r"^\d{3}[A-Z]$")


def _resolve_ocr_config_path(config_path: Path | None = None) -> Path:
    if config_path is not None:
        config_path = Path(config_path)
        if config_path.name == "ocr.yaml":
            return config_path

        candidate = config_path.parent / "ocr.yaml"
        if candidate.exists():
            return candidate

    return Path("configs/ocr.yaml")


def _load_ocr_config(config_path: Path | None = None) -> dict[str, Any]:
    resolved_path = _resolve_ocr_config_path(config_path)
    if not resolved_path.exists():
        return {}

    with resolved_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def _get_ocr_instance(config: dict[str, Any]) -> Any | None:
    global _OCR_INSTANCE

    if _OCR_INSTANCE is not None:
        return _OCR_INSTANCE
    if PaddleOCR is None:
        return None

    ocr_config = config.get("ocr", {})
    languages = ocr_config.get("languages", ["en"])
    lang = languages[0] if isinstance(languages, list) and languages else "en"
    use_angle_cls = bool(ocr_config.get("use_angle_cls", True))

    _OCR_INSTANCE = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang)
    return _OCR_INSTANCE


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


def _clip_bbox(bbox: list[object], width: int, height: int) -> tuple[int, int, int, int] | None:
    if len(bbox) != 4:
        return None

    try:
        x1, y1, x2, y2 = [int(float(value)) for value in bbox]
    except (TypeError, ValueError):
        return None

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def _passes_bbox_filter(bbox: tuple[int, int, int, int], image_shape: tuple[int, ...]) -> bool:
    image_height, image_width = image_shape[:2]
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    area = width * height

    if width < 20 or height < 20:
        return False
    if area < 1_500:
        return False
    if width > image_width * 0.9 or height > image_height * 0.95:
        return False
    return True


def _normalize_lot_candidates(texts: list[str]) -> list[str]:
    normalized: list[str] = []

    for text in texts:
        candidate = re.sub(r"[^A-Za-z0-9]", "", text.upper())
        if len(candidate) != 4:
            continue

        chars = list(candidate)

        for index in range(3):
            if chars[index] in {"O", "D", "Q"}:
                chars[index] = "0"
            elif chars[index] == "I":
                chars[index] = "1"
            elif chars[index] == "Z":
                chars[index] = "2"
            elif chars[index] == "S":
                chars[index] = "5"

        if chars[3] in {"8", "3"}:
            chars[3] = "B"
        elif chars[3] == "0":
            chars[3] = "D"

        candidate = "".join(chars)
        if _LOT_PATTERN.match(candidate):
            normalized.append(candidate)

    return normalized


def _extract_texts_from_ocr_result(result: Any, min_confidence: float) -> list[str]:
    texts: list[str] = []
    if not isinstance(result, list):
        return texts

    for line in result:
        if not isinstance(line, list):
            continue
        for word in line:
            if not isinstance(word, (list, tuple)) or len(word) < 2:
                continue
            payload = word[1]
            if not isinstance(payload, (list, tuple)) or len(payload) < 2:
                continue
            text = str(payload[0]).strip()
            confidence = float(payload[1])
            if text and confidence >= min_confidence:
                texts.append(text)
    return texts


def read_lot_information(
    image: dict[str, Any],
    label_detections: list[dict[str, object]],
    config_path: Path | None = None,
) -> list[dict[str, object]]:
    """Crop detected label boxes and read lot text with PaddleOCR."""
    raw_image = _load_raw_image(image)
    if raw_image is None or not label_detections:
        return []

    config = _load_ocr_config(config_path)
    ocr_model = _get_ocr_instance(config)
    if ocr_model is None:
        return []

    ocr_config = config.get("ocr", {})
    min_confidence = float(ocr_config.get("min_confidence", 0.5))
    image_height, image_width = raw_image.shape[:2]
    lot_records: list[dict[str, object]] = []

    for index, detection in enumerate(label_detections, start=1):
        bbox = detection.get("bbox")
        if not isinstance(bbox, list):
            continue

        clipped_bbox = _clip_bbox(bbox, image_width, image_height)
        if clipped_bbox is None or not _passes_bbox_filter(clipped_bbox, raw_image.shape):
            continue

        x1, y1, x2, y2 = clipped_bbox
        crop = raw_image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        result = ocr_model.ocr(crop, cls=True)
        texts = _extract_texts_from_ocr_result(result, min_confidence=min_confidence)
        lots = _normalize_lot_candidates(texts)
        if not lots:
            continue

        lot_records.append(
            {
                "label_index": index,
                "lot": lots[0],
                "bbox": [x1, y1, x2, y2],
                "raw_texts": texts,
                "detector_confidence": float(detection.get("confidence", 0.0)),
            }
        )

    return lot_records
