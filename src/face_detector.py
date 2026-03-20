from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

import cv2
import face_recognition
import numpy as np

FaceLocation = Tuple[int, int, int, int]


class FaceDetector:
    """Face detection and encoding wrapper around face_recognition."""

    def __init__(self, model: str = "hog", num_jitters: int = 1) -> None:
        self.model = model
        self.num_jitters = max(1, int(num_jitters))
        self.logger = logging.getLogger(self.__class__.__name__)

    def detect_faces(self, rgb_frame: np.ndarray) -> List[FaceLocation]:
        """Detect face bounding boxes from an RGB frame."""
        try:
            return face_recognition.face_locations(rgb_frame, model=self.model)
        except Exception as exc:  # pragma: no cover - defensive path
            self.logger.exception("Face detection failed: %s", exc)
            return []

    def encode_faces(self, rgb_frame: np.ndarray, face_locations: Sequence[FaceLocation]) -> List[np.ndarray]:
        """Compute 128-D face encodings for detected face locations."""
        if not face_locations:
            return []

        try:
            return face_recognition.face_encodings(
                rgb_frame,
                known_face_locations=list(face_locations),
                num_jitters=self.num_jitters,
            )
        except Exception as exc:  # pragma: no cover - defensive path
            self.logger.exception("Face encoding failed: %s", exc)
            return []

    @staticmethod
    def preprocess_frame(frame_bgr: np.ndarray, scale: float = 0.25) -> np.ndarray:
        """Resize and convert BGR frame to RGB for efficient face processing."""
        resized = cv2.resize(frame_bgr, (0, 0), fx=scale, fy=scale)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
