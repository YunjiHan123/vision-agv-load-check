from __future__ import annotations

from pathlib import Path

from src.pipeline.run_pipeline import run_pipeline


if __name__ == "__main__":
    print(run_pipeline(Path("data/samples"), Path("configs/pipeline.yaml")))
