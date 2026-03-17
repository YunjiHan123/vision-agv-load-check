from __future__ import annotations

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.evaluate import char_accuracy


def test_char_accuracy_exact_match() -> None:
    assert char_accuracy("A10001", "A10001") == 1.0


def test_char_accuracy_partial_match() -> None:
    assert char_accuracy("A10001", "A10099") == 4 / 6
