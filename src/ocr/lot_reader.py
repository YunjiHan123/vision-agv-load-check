from __future__ import annotations

import re
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from src.utils.lot_codes import LOT_POOL
from src.utils.text_distance import levenshtein_distance

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CACHE_ROOT_CANDIDATES = [
    Path(os.environ["BOOK_LOT_VISION_OCR_CACHE"]) if "BOOK_LOT_VISION_OCR_CACHE" in os.environ else None,
    Path("C:/temp/book_lot_vision_cache"),
    Path(tempfile.gettempdir()) / "book_lot_vision_cache",
    Path.home() / ".book_lot_vision_cache",
    _PROJECT_ROOT / "data" / "interim" / "ocr_cache",
]


def _choose_cache_root() -> Path:
    for candidate in _CACHE_ROOT_CANDIDATES:
        if candidate is None:
            continue
        try:
            test_dir = candidate / ".write_test"
            test_dir.mkdir(parents=True, exist_ok=True)
            test_dir.rmdir()
        except OSError:
            continue
        return candidate
    return _PROJECT_ROOT / "data" / "interim" / "ocr_cache"


_OCR_CACHE_ROOT = _choose_cache_root()
_PADDLE_CACHE_DIR = _OCR_CACHE_ROOT / "paddlex"
_PADDLE_HOME_DIR = _OCR_CACHE_ROOT / "home"
_PADDLE_DATA_HOME_DIR = _PADDLE_HOME_DIR / ".cache" / "paddle"
os.environ["HOME"] = str(_PADDLE_HOME_DIR)
os.environ["USERPROFILE"] = str(_PADDLE_HOME_DIR)
os.environ["XDG_CACHE_HOME"] = str(_PADDLE_HOME_DIR / ".cache")
os.environ["PADDLE_HOME"] = str(_PADDLE_DATA_HOME_DIR)
os.environ["PADDLE_PDX_CACHE_HOME"] = str(_PADDLE_CACHE_DIR)
for directory in (_PADDLE_CACHE_DIR, _PADDLE_HOME_DIR, _PADDLE_DATA_HOME_DIR):
    directory.mkdir(parents=True, exist_ok=True)

try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None

try:
    from easyocr import Reader as EasyOCRReader
except Exception:
    EasyOCRReader = None


_OCR_INSTANCE: Any | None = None
_EASYOCR_INSTANCE: Any | None = None
_LOT_PATTERN = re.compile(r"^\d{3}[A-Z]$")
_SIMILAR_DIGIT_GROUPS = [{"0", "8"}, {"1", "7"}, {"2", "3"}, {"5", "6"}]
_SIMILAR_SUFFIX_GROUPS = [{"A", "E"}, {"B", "D"}, {"C", "D"}]


def _resolve_runtime_cache_root(config: dict[str, Any]) -> Path:
    ocr_config = config.get("ocr", {})
    configured_root = ocr_config.get("cache_root")
    if isinstance(configured_root, str) and configured_root.strip():
        return Path(configured_root)
    return _OCR_CACHE_ROOT


def _apply_ocr_runtime_env(config: dict[str, Any]) -> Path:
    cache_root = _resolve_runtime_cache_root(config)
    paddle_cache_dir = cache_root / "paddlex"
    paddle_home_dir = cache_root / "home"
    paddle_data_home_dir = paddle_home_dir / ".cache" / "paddle"

    for directory in (paddle_cache_dir, paddle_home_dir, paddle_data_home_dir):
        directory.mkdir(parents=True, exist_ok=True)

    os.environ["HOME"] = str(paddle_home_dir)
    os.environ["USERPROFILE"] = str(paddle_home_dir)
    os.environ["XDG_CACHE_HOME"] = str(paddle_home_dir / ".cache")
    os.environ["PADDLE_HOME"] = str(paddle_data_home_dir)
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(paddle_cache_dir)
    return cache_root


def _resolve_ocr_config_path(config_path: Path | None = None) -> Path:
    if config_path is not None:
        config_path = Path(config_path)
        if config_path.is_file() and config_path.suffix.lower() in {".yaml", ".yml"} and "ocr" in config_path.stem.lower():
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

    _apply_ocr_runtime_env(config)

    ocr_config = config.get("ocr", {})
    languages = ocr_config.get("languages", ["en"])
    lang = languages[0] if isinstance(languages, list) and languages else "en"
    use_angle_cls = bool(ocr_config.get("use_angle_cls", True))
    ocr_version = str(ocr_config.get("ocr_version", "PP-OCRv4"))

    try:
        _OCR_INSTANCE = PaddleOCR(
            lang=lang,
            ocr_version=ocr_version,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=use_angle_cls,
        )
    except Exception:
        return None
    return _OCR_INSTANCE


def _get_easyocr_instance(config: dict[str, Any]) -> Any | None:
    global _EASYOCR_INSTANCE

    if _EASYOCR_INSTANCE is not None:
        return _EASYOCR_INSTANCE
    if EasyOCRReader is None:
        return None

    cache_root = _apply_ocr_runtime_env(config)

    ocr_config = config.get("ocr", {})
    languages = ocr_config.get("languages", ["en"])
    if not isinstance(languages, list) or not languages:
        languages = ["en"]

    model_storage_dir = cache_root / "easyocr_models"
    try:
        model_storage_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        model_storage_dir = _PROJECT_ROOT / "data" / "interim" / "ocr_cache" / "easyocr_models"
        model_storage_dir.mkdir(parents=True, exist_ok=True)

    try:
        _EASYOCR_INSTANCE = EasyOCRReader(
            languages,
            gpu=False,
            model_storage_directory=str(model_storage_dir),
            download_enabled=True,
            verbose=False,
        )
    except Exception:
        return None
    return _EASYOCR_INSTANCE


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


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    padding_ratio_x: float,
    padding_ratio_y: float,
    min_padding_px: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    box_width = x2 - x1
    box_height = y2 - y1
    pad_x = max(int(round(box_width * padding_ratio_x)), min_padding_px)
    pad_y = max(int(round(box_height * padding_ratio_y)), min_padding_px)

    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


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


def _tokenize_lot_text(text: str) -> list[str]:
    compact = re.sub(r"[^A-Za-z0-9]", "", text.upper())
    if len(compact) < 4:
        return [compact] if compact else []
    if len(compact) == 4:
        return [compact]
    return [compact[index : index + 4] for index in range(len(compact) - 3)]


def _repair_lot_candidate(candidate: str) -> str | None:
    if len(candidate) != 4:
        return None

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
    elif chars[3] == "4":
        chars[3] = "A"

    candidate = "".join(chars)
    return candidate if _LOT_PATTERN.match(candidate) else None


def _char_substitution_cost(left: str, right: str, position: int) -> float:
    if left == right:
        return 0.0

    if position < 3:
        if {left, right} in _SIMILAR_DIGIT_GROUPS:
            return 0.35
        return 1.0

    if {left, right} in _SIMILAR_SUFFIX_GROUPS:
        return 0.35
    return 1.0


def _lot_match_cost(candidate: str, valid_code: str) -> tuple[float, float, float]:
    prefix_cost = sum(_char_substitution_cost(candidate[index], valid_code[index], index) for index in range(3))
    suffix_cost = _char_substitution_cost(candidate[3], valid_code[3], 3)
    total_cost = prefix_cost + suffix_cost
    return total_cost, prefix_cost, suffix_cost


def _rank_valid_lot_matches(candidate: str, valid_lot_pool: list[str]) -> list[tuple[tuple[int, int, int], str]]:
    ranked: list[tuple[tuple[float, float, float], str]] = []
    for valid_code in valid_lot_pool:
        ranked.append((_lot_match_cost(candidate, valid_code), valid_code))
    return sorted(ranked, key=lambda item: (item[0], item[1]))


def _snap_to_valid_lot(
    candidate: str,
    valid_lot_pool: list[str],
) -> str | None:
    if candidate in valid_lot_pool:
        return candidate

    ranked_matches = _rank_valid_lot_matches(candidate, valid_lot_pool)
    if not ranked_matches:
        return None

    best_score, best_match = ranked_matches[0]
    if best_score[0] > 1.2 or best_score[1] > 1.0:
        return None

    if len(ranked_matches) > 1 and ranked_matches[1][0] == best_score:
        return None

    return best_match


def _normalize_lot_candidates(
    texts: list[str],
    valid_lot_pool: list[str] | None = None,
) -> list[str]:
    normalized: list[str] = []

    for text in texts:
        for token in _tokenize_lot_text(text):
            candidate = _repair_lot_candidate(token)
            if candidate is None:
                continue
            if valid_lot_pool:
                snapped_candidate = _snap_to_valid_lot(candidate, valid_lot_pool)
                if snapped_candidate is not None:
                    normalized.append(snapped_candidate)
                continue
            normalized.append(candidate)

    return normalized


def _collect_lot_votes(
    texts: list[str],
    valid_lot_pool: list[str],
) -> tuple[list[str], int, int]:
    direct_votes: list[str] = []
    snapped_votes: list[str] = []

    for text in texts:
        for token in _tokenize_lot_text(text):
            candidate = _repair_lot_candidate(token)
            if candidate is None:
                continue
            if candidate in valid_lot_pool:
                direct_votes.append(candidate)
                continue
            snapped_candidate = _snap_to_valid_lot(candidate, valid_lot_pool)
            if snapped_candidate is not None:
                snapped_votes.append(snapped_candidate)

    if direct_votes:
        return direct_votes, len(direct_votes), len(snapped_votes)
    return snapped_votes, 0, len(snapped_votes)


def _build_candidate_score_map(texts: list[str], valid_lot_pool: list[str]) -> dict[str, float]:
    candidate_scores: dict[str, float] = {}

    for text in texts:
        for token in _tokenize_lot_text(text):
            candidate = _repair_lot_candidate(token)
            if candidate is None:
                continue

            if candidate in valid_lot_pool:
                candidate_scores[candidate] = candidate_scores.get(candidate, 0.0) + 3.0
                continue

            ranked_matches = _rank_valid_lot_matches(candidate, valid_lot_pool)
            if not ranked_matches:
                continue

            best_score, best_match = ranked_matches[0]
            if best_score[0] > 1.35 or best_score[1] > 1.0:
                continue

            score = max(0.5, 2.0 - best_score[0])
            if len(ranked_matches) > 1 and ranked_matches[1][0] == best_score:
                score -= 0.4
            if score <= 0.0:
                continue
            candidate_scores[best_match] = candidate_scores.get(best_match, 0.0) + score

    return candidate_scores


def _extract_texts_from_ocr_result(result: Any, min_confidence: float) -> list[str]:
    texts: list[str] = []

    def _safe_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _visit(node: Any) -> None:
        if isinstance(node, dict):
            rec_texts = node.get("rec_texts")
            rec_scores = node.get("rec_scores")
            if isinstance(rec_texts, list):
                for index, text_value in enumerate(rec_texts):
                    text = str(text_value).strip()
                    score = 0.0
                    if isinstance(rec_scores, list) and index < len(rec_scores):
                        score = float(rec_scores[index])
                    if text and score >= min_confidence:
                        texts.append(text)
            if "rec_text" in node:
                text = str(node.get("rec_text", "")).strip()
                confidence = float(node.get("rec_score", 0.0))
                if text and confidence >= min_confidence:
                    texts.append(text)
            for value in node.values():
                _visit(value)
            return

        if isinstance(node, (list, tuple)):
            if (
                len(node) >= 3
                and isinstance(node[1], str)
                and not isinstance(node[2], (list, tuple, dict))
            ):
                text = node[1].strip()
                confidence = _safe_float(node[2])
                if text and confidence is not None and confidence >= min_confidence:
                    texts.append(text)
                    return
            if (
                len(node) >= 2
                and isinstance(node[1], (list, tuple))
                and len(node[1]) >= 2
                and not isinstance(node[1][1], (list, tuple, dict))
            ):
                text = str(node[1][0]).strip()
                confidence = _safe_float(node[1][1])
                if text and confidence is not None and confidence >= min_confidence:
                    texts.append(text)
                    return
            for item in node:
                _visit(item)

    _visit(result)
    return texts


def _rotate_image(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    if abs(angle_degrees) < 0.1:
        return image

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cos_value = abs(matrix[0, 0])
    sin_value = abs(matrix[0, 1])
    bound_width = int((height * sin_value) + (width * cos_value))
    bound_height = int((height * cos_value) + (width * sin_value))
    matrix[0, 2] += (bound_width / 2.0) - center[0]
    matrix[1, 2] += (bound_height / 2.0) - center[1]
    return cv2.warpAffine(
        image,
        matrix,
        (bound_width, bound_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _order_quad_points(points: np.ndarray) -> np.ndarray:
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    return np.array(
        [
            points[np.argmin(sums)],
            points[np.argmin(diffs)],
            points[np.argmax(sums)],
            points[np.argmax(diffs)],
        ],
        dtype=np.float32,
    )


def _extract_text_region_mask(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    kernel_width = max(3, crop.shape[1] // 18)
    kernel_height = max(3, crop.shape[0] // 10)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    connected = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    return cv2.morphologyEx(connected, cv2.MORPH_OPEN, open_kernel, iterations=1)


def _find_text_region_rect(crop: np.ndarray, min_area_ratio: float) -> tuple[tuple[float, float], tuple[float, float], float] | None:
    mask = _extract_text_region_mask(crop)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    crop_area = float(crop.shape[0] * crop.shape[1])
    candidate_points: list[np.ndarray] = []
    min_width = max(crop.shape[1] * 0.18, 12.0)
    min_height = max(crop.shape[0] * 0.18, 12.0)

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < crop_area * min_area_ratio:
            continue
        rect = cv2.minAreaRect(contour)
        (_, _), (width, height), _ = rect
        if max(width, height) < min_width or min(width, height) < min_height:
            continue
        candidate_points.append(contour.reshape(-1, 2))

    if not candidate_points:
        return None

    merged_points = np.concatenate(candidate_points).astype(np.float32)
    rect = cv2.minAreaRect(merged_points)
    (_, _), (width, height), _ = rect
    if width <= 1.0 or height <= 1.0:
        return None
    return rect


def _crop_rotated_rect(
    crop: np.ndarray,
    rect: tuple[tuple[float, float], tuple[float, float], float],
    padding_ratio: float,
) -> np.ndarray | None:
    (_, _), (width, height), _ = rect
    pad_width = width * (1.0 + padding_ratio)
    pad_height = height * (1.0 + padding_ratio)
    padded_rect = (rect[0], (pad_width, pad_height), rect[2])

    box = cv2.boxPoints(padded_rect)
    ordered_box = _order_quad_points(box)
    target_width = max(int(round(max(pad_width, pad_height))), 1)
    target_height = max(int(round(min(pad_width, pad_height))), 1)
    destination = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(ordered_box, destination)
    warped = cv2.warpPerspective(
        crop,
        matrix,
        (target_width, target_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if warped.size == 0:
        return None
    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped


def _refine_text_crop(crop: np.ndarray, ocr_config: dict[str, Any]) -> np.ndarray:
    if not bool(ocr_config.get("use_text_region_refine", True)):
        return crop

    min_area_ratio = float(ocr_config.get("text_region_min_area_ratio", 0.01))
    padding_ratio = float(ocr_config.get("text_region_padding_ratio", 0.08))
    rect = _find_text_region_rect(crop, min_area_ratio=min_area_ratio)
    if rect is None:
        return crop

    refined = _crop_rotated_rect(crop, rect, padding_ratio=padding_ratio)
    return refined if refined is not None else crop


def _estimate_text_skew_angle(crop: np.ndarray) -> float | None:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    points = cv2.findNonZero(thresholded)
    if points is None or len(points) < 20:
        return None

    angle = float(cv2.minAreaRect(points)[-1])
    if angle < -45.0:
        angle += 90.0
    if angle > 45.0:
        angle -= 90.0
    if abs(angle) > 25.0:
        return None
    return angle


def _resize_for_ocr(crop: np.ndarray, upscale_factor: float, min_width: int, min_height: int) -> np.ndarray:
    height, width = crop.shape[:2]
    scale = max(upscale_factor, min_width / max(width, 1), min_height / max(height, 1))
    if scale <= 1.0:
        return crop

    return cv2.resize(
        crop,
        (max(int(round(width * scale)), 1), max(int(round(height * scale)), 1)),
        interpolation=cv2.INTER_CUBIC,
    )


def _build_ocr_crops(
    crop: np.ndarray,
    ocr_config: dict[str, Any],
    *,
    use_text_region_refine: bool | None = None,
) -> list[np.ndarray]:
    use_deskew = bool(ocr_config.get("use_deskew", True))
    rotation_candidates = ocr_config.get("rotation_candidates", [-8, -4, 0, 4, 8])
    if not isinstance(rotation_candidates, list):
        rotation_candidates = [-8, -4, 0, 4, 8]

    min_width = int(ocr_config.get("min_crop_width", 160))
    min_height = int(ocr_config.get("min_crop_height", 48))
    upscale_factor = float(ocr_config.get("upscale_factor", 2.0))

    refine_enabled = bool(ocr_config.get("use_text_region_refine", True))
    if use_text_region_refine is not None:
        refine_enabled = use_text_region_refine

    refined = _refine_text_crop(crop, ocr_config) if refine_enabled else crop
    prepared = _resize_for_ocr(refined, upscale_factor=upscale_factor, min_width=min_width, min_height=min_height)
    crops = [prepared]

    if use_deskew:
        skew_angle = _estimate_text_skew_angle(prepared)
        if skew_angle is not None and abs(skew_angle) >= 1.0:
            crops.append(_rotate_image(prepared, -skew_angle))

    for angle in rotation_candidates:
        try:
            numeric_angle = float(angle)
        except (TypeError, ValueError):
            continue
        if abs(numeric_angle) < 0.1:
            continue
        crops.append(_rotate_image(prepared, numeric_angle))

    return crops


def _run_ocr_on_variants(
    crop_variants: list[np.ndarray],
    ocr_model: Any | None,
    easyocr_model: Any | None,
    ocr_config: dict[str, Any],
    min_confidence: float,
    valid_lot_pool: list[str],
) -> dict[str, Any]:
    texts: list[str] = []
    normalized_votes: list[str] = []
    direct_vote_count = 0
    snapped_vote_count = 0
    candidate_scores: dict[str, float] = {}

    for crop_variant in crop_variants:
        variant_texts: list[str] = []
        if ocr_model is not None:
            result = _run_ocr(ocr_model, crop_variant, use_angle_cls=bool(ocr_config.get("use_angle_cls", True)))
            variant_texts = _extract_texts_from_ocr_result(result, min_confidence=min_confidence)
        if not variant_texts and easyocr_model is not None:
            result = _run_easyocr(easyocr_model, crop_variant)
            variant_texts = _extract_texts_from_ocr_result(result, min_confidence=min_confidence)
        if not variant_texts:
            continue

        texts.extend(variant_texts)
        variant_votes, variant_direct_count, variant_snapped_count = _collect_lot_votes(
            variant_texts,
            valid_lot_pool=valid_lot_pool,
        )
        normalized_votes.extend(variant_votes)
        direct_vote_count += variant_direct_count
        snapped_vote_count += variant_snapped_count
        for candidate_code, score in _build_candidate_score_map(variant_texts, valid_lot_pool).items():
            candidate_scores[candidate_code] = candidate_scores.get(candidate_code, 0.0) + score

    texts = list(dict.fromkeys(texts))
    unique_lots = list(dict.fromkeys(normalized_votes))
    return {
        "texts": texts,
        "votes": normalized_votes,
        "unique_lots": unique_lots,
        "direct_vote_count": direct_vote_count,
        "snapped_vote_count": snapped_vote_count,
        "candidate_scores": candidate_scores,
    }


def _select_ocr_path_result(path_results: list[dict[str, Any]]) -> dict[str, Any]:
    available_results = [result for result in path_results if result.get("votes")]
    if not available_results:
        return {
            "texts": [],
            "votes": [],
            "unique_lots": [],
            "direct_vote_count": 0,
            "snapped_vote_count": 0,
            "candidate_scores": {},
        }

    return max(
        available_results,
        key=lambda result: (
            int(result.get("direct_vote_count", 0)),
            len(result.get("unique_lots", [])),
            -int(result.get("snapped_vote_count", 0)),
            len(result.get("candidate_scores", {})),
            len(result.get("texts", [])),
        ),
    )


def _candidate_entries_from_scores(candidate_scores: dict[str, float], top_k: int) -> list[dict[str, object]]:
    ranked_candidates = sorted(candidate_scores.items(), key=lambda item: (-item[1], item[0]))
    return [
        {"lot": lot_code, "score": round(score, 4)}
        for lot_code, score in ranked_candidates[:top_k]
        if score > 0.0
    ]


def _decode_lot_records(records: list[dict[str, object]], expected_count: int) -> list[dict[str, object]]:
    candidate_records = [record for record in records if record.get("candidate_lots")]
    if not candidate_records:
        return []

    resolved_records: list[dict[str, object]] = []
    used_lots: set[str] = set()
    duplicate_records: list[dict[str, object]] = []

    for record in candidate_records:
        candidate_lots = record.get("candidate_lots", [])
        if not isinstance(candidate_lots, list) or not candidate_lots:
            continue

        selected_entry = None
        for entry in candidate_lots:
            lot_code = entry.get("lot")
            if isinstance(lot_code, str) and lot_code not in used_lots:
                selected_entry = entry
                break

        if selected_entry is None:
            duplicate_records.append(record)
            selected_entry = candidate_lots[0]

        selected_lot = selected_entry.get("lot")
        if not isinstance(selected_lot, str):
            continue

        updated_record = dict(record)
        updated_record["lot"] = selected_lot
        updated_record["selected_score"] = float(selected_entry.get("score", 0.0))
        resolved_records.append(updated_record)
        used_lots.add(selected_lot)

    if expected_count <= 0:
        return resolved_records

    if len(resolved_records) > expected_count:
        resolved_records.sort(key=lambda item: float(item.get("selected_score", 0.0)), reverse=True)
        resolved_records = resolved_records[:expected_count]

    return resolved_records


def _complete_missing_lot_records(
    decoded_records: list[dict[str, object]],
    provisional_records: list[dict[str, object]],
    expected_count: int,
) -> list[dict[str, object]]:
    if expected_count <= 0 or len(decoded_records) >= expected_count:
        return decoded_records

    used_lots = {str(record.get("lot")) for record in decoded_records if record.get("lot")}
    supplemental_pool: list[tuple[float, str, dict[str, object]]] = []

    for record in provisional_records:
        candidate_lots = record.get("candidate_lots")
        if not isinstance(candidate_lots, list):
            continue

        for rank, entry in enumerate(candidate_lots):
            lot_code = entry.get("lot")
            if not isinstance(lot_code, str) or lot_code in used_lots:
                continue
            try:
                score = float(entry.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            adjusted_score = score - (rank * 0.35)
            if adjusted_score <= 0.0:
                continue
            supplemental_pool.append((adjusted_score, lot_code, record))

    supplemental_pool.sort(key=lambda item: (-item[0], item[1]))

    completed_records = list(decoded_records)
    for adjusted_score, lot_code, source_record in supplemental_pool:
        if len(completed_records) >= expected_count:
            break
        if lot_code in used_lots:
            continue

        completed_record = dict(source_record)
        completed_record["lot"] = lot_code
        completed_record["selected_score"] = adjusted_score
        completed_record["supplemental"] = True
        completed_records.append(completed_record)
        used_lots.add(lot_code)

    return completed_records


def _run_ocr(model: Any, crop: np.ndarray, use_angle_cls: bool) -> Any:
    try:
        return model.ocr(crop, cls=use_angle_cls)
    except TypeError:
        pass

    try:
        return model.ocr(crop)
    except TypeError:
        pass

    if hasattr(model, "predict"):
        return model.predict(crop, use_textline_orientation=use_angle_cls)

    return []


def _run_easyocr(model: Any, crop: np.ndarray) -> Any:
    try:
        return model.readtext(crop, detail=1)
    except Exception:
        return []


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
    use_easyocr_fallback = bool(config.get("ocr", {}).get("paddle_fallback_to_easyocr", True))
    easyocr_model = _get_easyocr_instance(config) if use_easyocr_fallback else None
    if ocr_model is None and easyocr_model is None:
        return []

    ocr_config = config.get("ocr", {})
    min_confidence = float(ocr_config.get("min_confidence", 0.5))
    valid_lot_pool = ocr_config.get("valid_lot_pool", LOT_POOL)
    if not isinstance(valid_lot_pool, list) or not valid_lot_pool:
        valid_lot_pool = LOT_POOL
    padding_ratio_x = float(ocr_config.get("crop_padding_ratio_x", ocr_config.get("crop_padding_ratio", 0.12)))
    padding_ratio_y = float(ocr_config.get("crop_padding_ratio_y", ocr_config.get("crop_padding_ratio", 0.18)))
    min_padding_px = int(ocr_config.get("crop_padding_pixels", 8))
    image_height, image_width = raw_image.shape[:2]
    provisional_records: list[dict[str, object]] = []
    expected_count = 0

    for index, detection in enumerate(label_detections, start=1):
        bbox = detection.get("bbox")
        if not isinstance(bbox, list):
            continue

        clipped_bbox = _clip_bbox(bbox, image_width, image_height)
        if clipped_bbox is None or not _passes_bbox_filter(clipped_bbox, raw_image.shape):
            continue
        expected_count += 1

        x1, y1, x2, y2 = clipped_bbox
        crop_bbox = _expand_bbox(
            clipped_bbox,
            image_width,
            image_height,
            padding_ratio_x=padding_ratio_x,
            padding_ratio_y=padding_ratio_y,
            min_padding_px=min_padding_px,
        )
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox
        crop = raw_image[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            continue

        path_results = [
            _run_ocr_on_variants(
                _build_ocr_crops(crop, ocr_config, use_text_region_refine=False),
                ocr_model=ocr_model,
                easyocr_model=easyocr_model,
                ocr_config=ocr_config,
                min_confidence=min_confidence,
                valid_lot_pool=valid_lot_pool,
            ),
            _run_ocr_on_variants(
                _build_ocr_crops(crop, ocr_config, use_text_region_refine=True),
                ocr_model=ocr_model,
                easyocr_model=easyocr_model,
                ocr_config=ocr_config,
                min_confidence=min_confidence,
                valid_lot_pool=valid_lot_pool,
            ),
        ]
        selected_result = _select_ocr_path_result(path_results)
        texts = selected_result["texts"]
        candidate_scores = selected_result.get("candidate_scores", {})
        if not isinstance(candidate_scores, dict) or not candidate_scores:
            continue
        candidate_lots = _candidate_entries_from_scores(
            candidate_scores,
            top_k=int(ocr_config.get("candidate_top_k", 3)),
        )
        if not candidate_lots:
            continue

        lots = selected_result["votes"]
        best_lot = str(candidate_lots[0]["lot"])

        provisional_records.append(
            {
                "label_index": index,
                "lot": best_lot,
                "bbox": [x1, y1, x2, y2],
                "crop_bbox": [crop_x1, crop_y1, crop_x2, crop_y2],
                "raw_texts": texts,
                "candidate_lots": candidate_lots,
                "vote_trace": list(dict.fromkeys(lots)),
                "detector_confidence": float(detection.get("confidence", 0.0)),
            }
        )

    decoded_records = _decode_lot_records(provisional_records, expected_count=expected_count)
    decoded_records = _complete_missing_lot_records(
        decoded_records,
        provisional_records,
        expected_count=expected_count,
    )
    decoded_records.sort(key=lambda item: int(item.get("label_index", 0)))
    return decoded_records
