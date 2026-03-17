from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline.run_pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the book lot vision pipeline.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/samples"),
        help="Input image file or directory.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pipeline.yaml"),
        help="Pipeline configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pipeline(input_path=args.input, config_path=args.config)
    print(result["summary"])


if __name__ == "__main__":
    main()
