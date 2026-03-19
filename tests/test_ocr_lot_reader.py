from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.ocr import lot_reader


def test_normalize_lot_candidates_repairs_common_ocr_confusions() -> None:
    candidates = lot_reader._normalize_lot_candidates(["o01b", "12-3d", "bad", "OO8B"])

    assert candidates == ["001B", "123D", "008B"]


def test_normalize_lot_candidates_snaps_to_valid_pool_and_extracts_embedded_tokens() -> None:
    candidates = lot_reader._normalize_lot_candidates(["lot:005e", "ref-0044"], valid_lot_pool=["004A", "004E"])

    assert candidates == ["004E", "004A"]


def test_normalize_lot_candidates_skips_ambiguous_snaps() -> None:
    candidates = lot_reader._normalize_lot_candidates(["001F"], valid_lot_pool=["001A", "001B"])

    assert candidates == []


def test_read_lot_information_returns_empty_without_ocr_dependency(monkeypatch) -> None:
    monkeypatch.setattr(lot_reader, "_OCR_INSTANCE", None)
    monkeypatch.setattr(lot_reader, "PaddleOCR", None)

    result = lot_reader.read_lot_information(
        image={"raw_image": np.zeros((120, 120, 3), dtype=np.uint8)},
        label_detections=[{"bbox": [10, 10, 90, 90], "confidence": 0.8}],
        config_path=Path("configs/pipeline.yaml"),
    )

    assert result == []


def test_read_lot_information_extracts_lot_from_mocked_ocr(monkeypatch) -> None:
    class FakeOCR:
        def ocr(self, crop: np.ndarray, cls: bool = True):  # noqa: ARG002
            if crop.shape[1] < 150:
                return []
            return [[[[0, 0], ("o01b", 0.91)], [[0, 0], ("noise", 0.2)]]]

    monkeypatch.setattr(lot_reader, "_OCR_INSTANCE", FakeOCR())
    monkeypatch.setattr(lot_reader, "_EASYOCR_INSTANCE", None)
    monkeypatch.setattr(lot_reader, "PaddleOCR", object())
    monkeypatch.setattr(lot_reader, "EasyOCRReader", None)

    result = lot_reader.read_lot_information(
        image={"raw_image": np.zeros((160, 160, 3), dtype=np.uint8)},
        label_detections=[{"bbox": [20, 20, 120, 120], "confidence": 0.77}],
        config_path=Path("configs/pipeline.yaml"),
    )

    assert len(result) == 1
    assert result[0]["label_index"] == 1
    assert result[0]["lot"] == "001B"
    assert result[0]["bbox"] == [20, 20, 120, 120]
    assert result[0]["crop_bbox"] == [4, 0, 136, 142]
    assert result[0]["raw_texts"] == ["o01b"]
    assert result[0]["candidate_lots"][0]["lot"] == "001B"
    assert result[0]["vote_trace"] == ["001B"]
    assert result[0]["detector_confidence"] == 0.77
    assert float(result[0]["selected_score"]) > 0.0


def test_read_lot_information_falls_back_to_easyocr(monkeypatch) -> None:
    class FakeEasyOCR:
        def readtext(self, crop: np.ndarray, detail: int = 1):  # noqa: ARG002
            return [([[1, 2], [3, 4]], "004e", 0.93)]

    monkeypatch.setattr(lot_reader, "_OCR_INSTANCE", None)
    monkeypatch.setattr(lot_reader, "_EASYOCR_INSTANCE", FakeEasyOCR())
    monkeypatch.setattr(lot_reader, "PaddleOCR", None)
    monkeypatch.setattr(lot_reader, "EasyOCRReader", object())

    result = lot_reader.read_lot_information(
        image={"raw_image": np.zeros((160, 160, 3), dtype=np.uint8)},
        label_detections=[{"bbox": [20, 20, 120, 120], "confidence": 0.88}],
        config_path=Path("configs/pipeline.yaml"),
    )

    assert result[0]["lot"] == "004E"
    assert result[0]["raw_texts"] == ["004e"]
    assert result[0]["candidate_lots"][0]["lot"] == "004E"


def test_extract_texts_from_paddleocr_v3_result() -> None:
    result = [
        {
            "rec_texts": ["004E", "bad"],
            "rec_scores": [0.96, 0.2],
        }
    ]

    assert lot_reader._extract_texts_from_ocr_result(result, min_confidence=0.5) == ["004E"]


def test_extract_texts_ignores_non_numeric_confidence_slots() -> None:
    result = [["meta", "002A", "raw_text"], {"rec_text": "004E", "rec_score": 0.91}]

    assert lot_reader._extract_texts_from_ocr_result(result, min_confidence=0.5) == ["004E"]


def test_build_ocr_crops_adds_rotated_variants() -> None:
    crop = np.zeros((60, 120, 3), dtype=np.uint8)

    crops = lot_reader._build_ocr_crops(
        crop,
        {
            "use_deskew": False,
            "rotation_candidates": [-4, 0, 4],
            "upscale_factor": 1.5,
            "min_crop_width": 120,
            "min_crop_height": 60,
        },
    )

    assert len(crops) == 3
    assert crops[0].shape[1] >= 120


def test_refine_text_crop_rectifies_rotated_text_region() -> None:
    crop = np.full((120, 180, 3), 255, dtype=np.uint8)
    box = cv2.boxPoints(((90.0, 60.0), (90.0, 28.0), 24.0)).astype(np.int32)
    cv2.fillConvexPoly(crop, box, (0, 0, 0))

    refined = lot_reader._refine_text_crop(
        crop,
        {
            "use_text_region_refine": True,
            "text_region_min_area_ratio": 0.01,
            "text_region_padding_ratio": 0.05,
        },
    )

    assert refined.shape[1] > refined.shape[0]
    assert refined.shape[0] < crop.shape[0]


def test_collect_lot_votes_prefers_direct_pool_hits_over_snapped_hits() -> None:
    votes, direct_count, snapped_count = lot_reader._collect_lot_votes(["001D", "002D"], valid_lot_pool=["001D", "002D"])

    assert votes == ["001D", "002D"]
    assert direct_count == 2
    assert snapped_count == 0


def test_select_ocr_path_result_prefers_more_direct_hits() -> None:
    result = lot_reader._select_ocr_path_result(
        [
            {
                "texts": ["001D"],
                "votes": ["001D"],
                "unique_lots": ["001D"],
                "direct_vote_count": 1,
                "snapped_vote_count": 0,
            },
            {
                "texts": ["002D", "001D"],
                "votes": ["002D", "001D"],
                "unique_lots": ["002D", "001D"],
                "direct_vote_count": 2,
                "snapped_vote_count": 1,
            },
        ]
    )

    assert result["unique_lots"] == ["002D", "001D"]


def test_decode_lot_records_avoids_duplicate_top_choice_when_alternative_exists() -> None:
    decoded = lot_reader._decode_lot_records(
        [
            {
                "label_index": 1,
                "candidate_lots": [{"lot": "004C", "score": 3.0}, {"lot": "004D", "score": 2.5}],
            },
            {
                "label_index": 2,
                "candidate_lots": [{"lot": "004C", "score": 2.9}, {"lot": "004A", "score": 2.7}],
            },
        ],
        expected_count=2,
    )

    assert [record["lot"] for record in decoded] == ["004C", "004A"]


def test_complete_missing_lot_records_adds_best_unused_candidate() -> None:
    decoded = [
        {
            "label_index": 1,
            "lot": "001D",
            "candidate_lots": [{"lot": "001D", "score": 3.0}, {"lot": "002D", "score": 2.4}],
            "selected_score": 3.0,
        },
        {
            "label_index": 2,
            "lot": "004B",
            "candidate_lots": [{"lot": "004B", "score": 2.8}, {"lot": "004A", "score": 2.6}],
            "selected_score": 2.8,
        },
        {
            "label_index": 3,
            "lot": "003C",
            "candidate_lots": [{"lot": "003C", "score": 2.7}, {"lot": "004A", "score": 2.5}],
            "selected_score": 2.7,
        },
    ]

    completed = lot_reader._complete_missing_lot_records(decoded, decoded, expected_count=4)

    assert len(completed) == 4
    assert {record["lot"] for record in completed} == {"001D", "004B", "003C", "004A"}
