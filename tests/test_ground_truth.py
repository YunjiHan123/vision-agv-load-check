from __future__ import annotations

import csv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
GROUND_TRUTH_CSV = BASE_DIR / "data" / "ground_truth.csv"
LOT_POOL = {
    "001A",
    "001B",
    "001C",
    "001D",
    "001E",
    "002A",
    "002B",
    "002C",
    "002D",
    "002E",
    "003A",
    "003B",
    "003C",
    "003D",
    "003E",
    "004A",
    "004B",
    "004C",
    "004D",
    "004E",
}


def load_ground_truth_rows() -> list[dict[str, str]]:
    with GROUND_TRUTH_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def split_lots(lot_text: str) -> list[str]:
    return [token.strip() for token in lot_text.split("|") if token.strip()]


def test_ground_truth_rows_match_current_dataset_rules() -> None:
    rows = load_ground_truth_rows()

    assert rows

    for row in rows:
        true_count = int(row["true_count"])
        lots = split_lots(row["true_lots"])

        assert 1 <= true_count <= 4
        assert len(lots) == true_count
        assert row["true_status"] in {"normal", "misaligned", "fallen"}
        assert row["condition"] in {"clear", "blur", "glare", "partial_occlusion"}
        assert all(lot in LOT_POOL for lot in lots)


def test_ground_truth_single_book_rows_are_normal() -> None:
    rows = load_ground_truth_rows()
    single_book_rows = [row for row in rows if int(row["true_count"]) == 1]

    assert single_book_rows
    assert all(row["true_status"] == "normal" for row in single_book_rows)


def test_ground_truth_image_names_are_unique_even_if_indices_are_sparse() -> None:
    rows = load_ground_truth_rows()
    image_names = [row["image_name"] for row in rows]

    assert len(image_names) == len(set(image_names))
    assert all(name.startswith("img_") and name.endswith(".jpg") for name in image_names)
