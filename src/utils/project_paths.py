from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIGS_DIR = ROOT_DIR / "configs"
DATA_DIR = ROOT_DIR / "data"

GROUND_TRUTH_CSV = DATA_DIR / "ground_truth.csv"
RESULT_CSV = DATA_DIR / "results.csv"
FAILURE_CSV = DATA_DIR / "failure_cases.csv"
PLOTS_DIR = DATA_DIR / "plots"
TEST_IMAGES_DIR = DATA_DIR / "test_images"

PIPELINE_CONFIG_PATH = CONFIGS_DIR / "pipeline.yaml"
