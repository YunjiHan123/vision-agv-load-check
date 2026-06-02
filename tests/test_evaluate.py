from __future__ import annotations

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.evaluate import char_accuracy
from src.utils.status import normalize_status_label


def test_char_accuracy_exact_match() -> None:
    assert char_accuracy("001A|001B", "001A|001B") == 1.0


def test_char_accuracy_partial_match() -> None:
    assert char_accuracy("001A|001B", "001A|001C") == 8 / 9


def test_status_normalization_merges_shifted_and_tilted() -> None:
    assert normalize_status_label("shifted") == "misaligned"
    assert normalize_status_label("tilted") == "misaligned"
    assert normalize_status_label("fallen") == "fallen"
