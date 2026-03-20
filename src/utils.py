from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Union

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
ENCODINGS_PATH = PROJECT_ROOT / "encodings" / "encodings.pkl"
DATABASE_PATH = PROJECT_ROOT / "database" / "attendance.db"
REPORT_PATH = PROJECT_ROOT / "reports" / "attendance_report.csv"


def ensure_directories(paths: Iterable[Union[str, Path]]) -> None:
    """Create directories for each path.

    If a file path is passed, its parent directory is created.
    """
    for path in paths:
        path_obj = Path(path)
        target = path_obj if path_obj.suffix == "" else path_obj.parent
        target.mkdir(parents=True, exist_ok=True)


def configure_logging(level: str = "INFO", log_file: Union[str, Path, None] = None) -> None:
    """Configure root logger for console and optional file logging."""
    handlers = [logging.StreamHandler()]

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def resolve_video_source(source: str) -> Union[int, str]:
    """Resolve video source from CLI input.

    Numeric values are treated as webcam indices.
    Non-numeric values are treated as file paths.
    """
    source = str(source).strip()
    if source.isdigit():
        return int(source)

    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Video file not found: {source_path}")
    return str(source_path)
