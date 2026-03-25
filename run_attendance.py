from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.attendance_manager import AttendanceManager
from src.face_detector import FaceDetector
from src.recognizer import FaceRecognizer
from src.utils import DATABASE_PATH, ENCODINGS_PATH, REPORT_PATH, configure_logging, ensure_directories, resolve_video_source
from src.video_processor import VideoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run automated attendance using webcam/video stream")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: webcam index (e.g. 0) or path to video file",
    )
    parser.add_argument("--encodings-path", type=Path, default=ENCODINGS_PATH, help="Path to trained encodings file")
    parser.add_argument("--database-path", type=Path, default=DATABASE_PATH, help="Path to SQLite database")
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH, help="Output CSV report path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Recognition distance threshold")
    parser.add_argument(
        "--detection-model",
        type=str,
        default="hog",
        choices=["hog", "cnn"],
        help="Face detection backend",
    )
    parser.add_argument(
        "--process-every-n",
        type=int,
        default=2,
        help="Process every Nth frame for speed optimization",
    )
    parser.add_argument(
        "--resize-scale",
        type=float,
        default=0.25,
        help="Frame resize scale before face detection",
    )
    parser.add_argument(
        "--recognition-cooldown",
        type=int,
        default=30,
        help="Frames to wait before re-recognizing same track",
    )
    parser.add_argument(
        "--tracker-max-disappeared",
        type=int,
        default=20,
        help="Tracker tolerance for missing detections",
    )
    parser.add_argument(
        "--tracker-max-distance",
        type=float,
        default=80.0,
        help="Max centroid distance for track association",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(level=args.log_level)
    logger = logging.getLogger("run_attendance")

    try:
        ensure_directories([args.database_path, args.report_path])

        # Prompt for session info
        session_name = input("Enter session name: ").strip()
        while not session_name:
            session_name = input("Session name cannot be empty. Enter session name: ").strip()
        while True:
            try:
                session_duration = int(input("Enter session duration in seconds: "))
                if session_duration > 0:
                    break
                print("Duration must be a positive integer.")
            except ValueError:
                print("Please enter a valid integer for duration.")

        source = resolve_video_source(args.source)
        detector = FaceDetector(model=args.detection_model)
        recognizer = FaceRecognizer(encodings_path=args.encodings_path, threshold=args.threshold)

        with AttendanceManager(db_path=args.database_path) as attendance_manager:
            video_processor = VideoProcessor(
                face_detector=detector,
                face_recognizer=recognizer,
                attendance_manager=attendance_manager,
                process_every_n=args.process_every_n,
                resize_scale=args.resize_scale,
                recognition_cooldown_frames=args.recognition_cooldown,
                tracker_max_disappeared=args.tracker_max_disappeared,
                tracker_max_distance=args.tracker_max_distance,
            )
            video_processor.run(
                source=source,
                report_path=args.report_path,
                session_name=session_name,
                session_duration=session_duration,
            )

        logger.info("Attendance session ended")
        logger.info("Report saved at: %s", args.report_path)
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as exc:
        logger.exception("Attendance run failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
