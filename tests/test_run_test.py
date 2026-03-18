from __future__ import annotations

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.run_test import build_mock_prediction


def test_mock_predictions_include_normal_and_abnormal_outputs() -> None:
    assert build_mock_prediction("img_001.jpg")["pred_status"] == "normal"
    assert build_mock_prediction("img_045.jpg")["pred_status"] == "misaligned"
    assert build_mock_prediction("img_009.jpg")["pred_status"] == "normal"
