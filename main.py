import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from disease_prediction.config import PipelineConfig
from disease_prediction.pipeline import run_training_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train heart disease prediction models.")
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to heart disease CSV file (for example: data/heart.csv).",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory where model and reports are saved.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = parse_args()

    config = PipelineConfig(
        data_path=args.data_path,
        artifacts_dir=args.artifacts_dir,
        model_path=args.artifacts_dir / "model.pkl",
        report_path=args.artifacts_dir / "metrics.json",
        feature_importance_path=args.artifacts_dir / "feature_importance.csv",
    )

    run_training_pipeline(config)
    logging.info("Training finished. Artifacts saved under: %s", config.artifacts_dir)


if __name__ == "__main__":
    main()

