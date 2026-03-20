from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.face_encoder import FaceEncoder
from src.utils import DATASET_DIR, ENCODINGS_PATH, configure_logging, ensure_directories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train face encodings from dataset images")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR, help="Path to student image dataset")
    parser.add_argument(
        "--encodings-path",
        type=Path,
        default=ENCODINGS_PATH,
        help="Path to save generated encodings (.pkl)",
    )
    parser.add_argument(
        "--detection-model",
        type=str,
        default="hog",
        choices=["hog", "cnn"],
        help="Face detection model to use during training",
    )
    parser.add_argument(
        "--num-jitters",
        type=int,
        default=1,
        help="Number of re-sampling steps for encoding generation",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(level=args.log_level)
    logger = logging.getLogger("train_model")

    ensure_directories([args.encodings_path])

    try:
        encoder = FaceEncoder(model=args.detection_model, num_jitters=args.num_jitters)
        summary = encoder.train(args.dataset_dir, args.encodings_path)

        logger.info("Training completed successfully")
        logger.info("Total images: %d", summary.total_images)
        logger.info("Processed images: %d", summary.processed_images)
        logger.info("Skipped images: %d", summary.skipped_images)
        logger.info("Students found: %d", summary.students_found)
        logger.info("Encodings saved: %d", summary.encodings_saved)
        return 0
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
