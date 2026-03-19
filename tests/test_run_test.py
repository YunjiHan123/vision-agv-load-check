from __future__ import annotations

import csv
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.run_test import RESULT_CSV, build_mock_prediction, generate_mock_results, load_ground_truth_rows
import src.run_test as run_test_module


def test_mock_predictions_include_normal_and_abnormal_outputs() -> None:
    assert build_mock_prediction("img_001.jpg")["pred_status"] == "normal"
    assert build_mock_prediction("img_045.jpg")["pred_status"] == "misaligned"
    assert build_mock_prediction("img_009.jpg")["pred_status"] == "normal"


def test_mock_predictions_follow_sparse_ground_truth_rows() -> None:
    existing_rows = load_ground_truth_rows(BASE_DIR / "data" / "ground_truth.csv")

    for row in existing_rows[:5]:
        prediction = build_mock_prediction(row["image_name"])
        assert "pred_lots" in prediction
        assert isinstance(prediction["pred_lots"], str)


def test_generate_mock_results_writes_one_result_per_ground_truth_row() -> None:
    expected_count = len(load_ground_truth_rows(BASE_DIR / "data" / "ground_truth.csv"))

    processed_count = generate_mock_results()

    assert processed_count == expected_count

    with RESULT_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == expected_count
    assert set(rows[0].keys()) == {"image_name", "pred_count", "pred_status", "pred_lots"}


def test_generate_real_results_writes_ocr_lots(monkeypatch, tmp_path) -> None:
    output_csv = tmp_path / "results.csv"

    def fake_process_image(image_path: Path, config_path: Path) -> dict[str, str | int]:
        assert config_path == run_test_module.CONFIG_PATH
        return {
            "pred_count": 2,
            "pred_status": "normal",
            "pred_lots": f"{image_path.stem.upper()}|001B",
        }

    monkeypatch.setattr(run_test_module, "RESULT_CSV", output_csv)
    monkeypatch.setattr(run_test_module, "process_image", fake_process_image)

    image_files = [tmp_path / "img_101.jpg", tmp_path / "img_102.jpg"]
    for image_path in image_files:
        image_path.write_bytes(b"placeholder")

    processed_count = run_test_module.generate_real_results(image_files)

    with output_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert processed_count == 2
    assert [row["pred_lots"] for row in rows] == ["IMG_101|001B", "IMG_102|001B"]
