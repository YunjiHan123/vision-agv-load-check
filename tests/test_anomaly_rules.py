from __future__ import annotations

from src.anomaly.rules import classify_status_from_detections


def _detection(x1: int, y1: int, x2: int, y2: int) -> dict[str, object]:
    return {"bbox": [x1, y1, x2, y2]}


def test_classify_normal_stack_from_aligned_boxes() -> None:
    detections = [
        _detection(100, 20, 170, 120),
        _detection(102, 120, 172, 220),
        _detection(98, 220, 168, 320),
        _detection(101, 320, 171, 420),
    ]

    status, metrics = classify_status_from_detections(detections)

    assert status == "normal"
    assert metrics["wide_box_ratio"] == 0.0


def test_classify_misaligned_stack_from_centerline_drift() -> None:
    detections = [
        _detection(100, 20, 170, 120),
        _detection(122, 120, 192, 220),
        _detection(88, 220, 158, 320),
        _detection(116, 320, 186, 420),
    ]

    status, metrics = classify_status_from_detections(detections)

    assert status == "misaligned"
    assert metrics["center_x_span"] > 20.0


def test_classify_fallen_stack_from_wide_boxes() -> None:
    detections = [
        _detection(20, 180, 140, 250),
        _detection(120, 150, 240, 230),
        _detection(220, 130, 340, 220),
        _detection(330, 170, 470, 250),
    ]

    status, metrics = classify_status_from_detections(detections)

    assert status == "fallen"
    assert metrics["wide_box_ratio"] >= 0.5
