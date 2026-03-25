from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .attendance_manager import AttendanceManager
from .face_detector import FaceDetector
from .recognizer import FaceRecognizer

FaceLocation = Tuple[int, int, int, int]


@dataclass
class Track:
    object_id: int
    centroid: Tuple[int, int]
    bbox: FaceLocation
    disappeared: int = 0
    name: str = "Unknown"
    confidence: float = 0.0
    distance: float = 1.0
    last_recognition_frame: int = -100000


class CentroidTracker:
    """Simple centroid tracker to maintain identity across nearby frames."""

    def __init__(self, max_disappeared: int = 20, max_distance: float = 80.0) -> None:
        self.max_disappeared = max(1, int(max_disappeared))
        self.max_distance = float(max_distance)
        self.next_object_id = 0
        self.tracks: Dict[int, Track] = {}

    @staticmethod
    def _centroid(bbox: FaceLocation) -> Tuple[int, int]:
        top, right, bottom, left = bbox
        c_x = int((left + right) / 2)
        c_y = int((top + bottom) / 2)
        return c_x, c_y

    def _register(self, bbox: FaceLocation) -> int:
        object_id = self.next_object_id
        self.next_object_id += 1

        self.tracks[object_id] = Track(
            object_id=object_id,
            centroid=self._centroid(bbox),
            bbox=bbox,
        )
        return object_id

    def _deregister(self, object_id: int) -> None:
        self.tracks.pop(object_id, None)

    def update(self, detections: List[FaceLocation]) -> Dict[int, int]:
        """Update tracker with detections.

        Returns:
            Dict[int, int]: Mapping from detection index to object_id.
        """
        detection_to_object: Dict[int, int] = {}

        if len(detections) == 0:
            stale_ids = []
            for object_id, track in self.tracks.items():
                track.disappeared += 1
                if track.disappeared > self.max_disappeared:
                    stale_ids.append(object_id)

            for object_id in stale_ids:
                self._deregister(object_id)

            return detection_to_object

        input_centroids = np.array([self._centroid(bbox) for bbox in detections], dtype=np.float32)

        if len(self.tracks) == 0:
            for det_idx, bbox in enumerate(detections):
                object_id = self._register(bbox)
                detection_to_object[det_idx] = object_id
            return detection_to_object

        object_ids = list(self.tracks.keys())
        object_centroids = np.array([self.tracks[obj_id].centroid for obj_id in object_ids], dtype=np.float32)

        distance_matrix = np.linalg.norm(
            object_centroids[:, np.newaxis, :] - input_centroids[np.newaxis, :, :],
            axis=2,
        )

        rows = distance_matrix.min(axis=1).argsort()
        cols = distance_matrix.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            if distance_matrix[row, col] > self.max_distance:
                continue

            object_id = object_ids[row]
            track = self.tracks[object_id]
            track.centroid = tuple(input_centroids[col].astype(int))
            track.bbox = detections[col]
            track.disappeared = 0

            detection_to_object[col] = object_id
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(distance_matrix.shape[0])) - used_rows
        unused_cols = set(range(distance_matrix.shape[1])) - used_cols

        for row in unused_rows:
            object_id = object_ids[row]
            track = self.tracks[object_id]
            track.disappeared += 1
            if track.disappeared > self.max_disappeared:
                self._deregister(object_id)

        for col in unused_cols:
            object_id = self._register(detections[col])
            detection_to_object[col] = object_id

        return detection_to_object


class VideoProcessor:
    """Main runtime engine for detection, tracking, recognition, and attendance marking."""

    def __init__(
        self,
        face_detector: FaceDetector,
        face_recognizer: FaceRecognizer,
        attendance_manager: AttendanceManager,
        process_every_n: int = 2,
        resize_scale: float = 0.25,
        recognition_cooldown_frames: int = 30,
        tracker_max_disappeared: int = 20,
        tracker_max_distance: float = 80.0,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.face_detector = face_detector
        self.face_recognizer = face_recognizer
        self.attendance_manager = attendance_manager

        self.process_every_n = max(1, int(process_every_n))
        self.resize_scale = float(resize_scale)
        if self.resize_scale <= 0.0 or self.resize_scale > 1.0:
            raise ValueError("resize_scale must be in (0.0, 1.0]")
        self.inverse_scale = 1.0 / self.resize_scale
        self.recognition_cooldown_frames = max(1, int(recognition_cooldown_frames))

        self.tracker = CentroidTracker(
            max_disappeared=tracker_max_disappeared,
            max_distance=tracker_max_distance,
        )

    def _scale_locations(self, face_locations: List[FaceLocation]) -> List[FaceLocation]:
        scaled_locations: List[FaceLocation] = []
        for top, right, bottom, left in face_locations:
            scaled_locations.append(
                (
                    int(top * self.inverse_scale),
                    int(right * self.inverse_scale),
                    int(bottom * self.inverse_scale),
                    int(left * self.inverse_scale),
                )
            )
        return scaled_locations

    @staticmethod
    def _draw_label(frame: np.ndarray, bbox: FaceLocation, label: str, color: Tuple[int, int, int]) -> None:
        top, right, bottom, left = bbox
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 24), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    def _draw_tracks(self, frame: np.ndarray) -> None:
        for track in self.tracker.tracks.values():
            if track.disappeared > self.tracker.max_disappeared // 2:
                continue

            is_known = track.name != "Unknown"
            color = (0, 180, 0) if is_known else (0, 0, 220)
            if is_known:
                label = f"{track.name} {track.confidence * 100:.1f}%"
            else:
                label = "Unknown"

            self._draw_label(frame, track.bbox, label, color)

    def run(
        self,
        source: Union[int, str] = 0,
        window_name: str = "Automated Attendance",
        report_path: Optional[Path] = None,
        session_name: Optional[str] = None,
        session_duration: Optional[int] = None,
    ) -> None:
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video source: {source}")

        self.logger.info("Starting video stream from source: %s", source)
        frame_count = 0
        fps_timer = time.time()
        fps_value = 0.0

        # Session attendance tracking
        if session_duration is None or session_duration <= 0:
            session_duration = 60  # default 60s if not provided
        if not session_name:
            session_name = "default_session"


        session_start_time = time.time()
        session_end_time = session_start_time + session_duration
        successful_detections_per_student = {}  # name -> count of successful detections
        total_detection_attempts = 0  # total number of detection attempts (frames where detection is performed)
        all_detected_students = set()
        present_students = set()

        try:
            while True:
                now = time.time()
                if now >= session_end_time:
                    self.logger.info("Session duration elapsed. Stopping attendance.")
                    break

                ret, frame = capture.read()
                if not ret:
                    self.logger.info("End of video stream or frame capture failed")
                    break

                frame_count += 1


                detected_this_frame = set()

                if frame_count % self.process_every_n == 0:
                    total_detection_attempts += 1
                    rgb_small = self.face_detector.preprocess_frame(frame, self.resize_scale)
                    locations_small = self.face_detector.detect_faces(rgb_small)
                    face_encodings = self.face_detector.encode_faces(rgb_small, locations_small)
                    locations_full = self._scale_locations(locations_small)

                    detection_map = self.tracker.update(locations_full)

                    for det_idx, object_id in detection_map.items():
                        track = self.tracker.tracks.get(object_id)
                        if track is None:
                            continue

                        # Known tracks are not repeatedly re-recognized.
                        if track.name != "Unknown":
                            detected_this_frame.add(track.name)
                            continue

                        if frame_count - track.last_recognition_frame < self.recognition_cooldown_frames:
                            continue

                        if det_idx >= len(face_encodings):
                            continue

                        result = self.face_recognizer.recognize_encoding(face_encodings[det_idx])
                        track.name = result.name
                        track.distance = result.distance
                        track.confidence = result.confidence
                        track.last_recognition_frame = frame_count

                        if result.is_known:
                            detected_this_frame.add(result.name)
                            all_detected_students.add(result.name)
                        else:
                            self.logger.info("Unknown face detected")

                # Update successful detections for each student detected in this frame
                for name in detected_this_frame:
                    if name == "Unknown":
                        continue
                    successful_detections_per_student.setdefault(name, 0)
                    successful_detections_per_student[name] += 1
                    all_detected_students.add(name)

                self._draw_tracks(frame)

                elapsed = max(1e-6, time.time() - fps_timer)
                fps_value = (fps_value * 0.9) + ((1.0 / elapsed) * 0.1)
                fps_timer = time.time()
                cv2.putText(frame, f"FPS: {fps_value:.1f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.logger.info("Exit requested by user")
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()

            # After session, mark attendance for all students
            # Fetch all students from DB

            db_students = self.attendance_manager.get_all_students()
            detection_threshold = 0.75  # 75% of detection attempts
            for name in db_students:
                detections = successful_detections_per_student.get(name, 0)
                percent = (detections / total_detection_attempts * 100) if total_detection_attempts > 0 else 0.0
                status = "Present" if percent >= (detection_threshold * 100) else "Absent"
                self.attendance_manager.mark_attendance(
                    student_name=name,
                    status=status,
                    session_name=session_name,
                    duration=percent,  # Store detection percentage in duration column for reporting
                )
                self.logger.info(f"Session '{session_name}': {name} - {status} ({percent:.1f}% detection)")

            if report_path is not None:
                self.attendance_manager.generate_report(report_path)
