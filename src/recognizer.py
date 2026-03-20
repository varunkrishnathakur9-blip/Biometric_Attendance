from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class RecognitionResult:
    name: str
    distance: float
    confidence: float
    is_known: bool


class FaceRecognizer:
    """Face recognizer backed by sklearn NearestNeighbors for fast matching."""

    def __init__(self, encodings_path: Path, threshold: float = 0.5) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.encodings_path = Path(encodings_path)
        self.threshold = float(threshold)

        self.known_encodings: np.ndarray
        self.known_names: List[str]
        self.nn_model: NearestNeighbors

        self._load_encodings()
        self._build_index()

    def _load_encodings(self) -> None:
        if not self.encodings_path.exists():
            raise FileNotFoundError(f"Encodings file not found: {self.encodings_path}")

        with self.encodings_path.open("rb") as file:
            payload = pickle.load(file)

        encodings = payload.get("encodings")
        names = payload.get("names")

        if encodings is None or names is None:
            raise ValueError("Encodings file missing required keys: 'encodings', 'names'")
        if len(encodings) == 0:
            raise ValueError("Encodings file is empty. Run training first.")

        self.known_encodings = np.asarray(encodings, dtype=np.float32)
        self.known_names = list(names)

        if self.known_encodings.shape[0] != len(self.known_names):
            raise ValueError("Mismatch between number of encodings and names.")

        self.logger.info("Loaded %d encodings from %s", len(self.known_names), self.encodings_path)

    def _build_index(self) -> None:
        self.nn_model = NearestNeighbors(n_neighbors=1, metric="euclidean")
        self.nn_model.fit(self.known_encodings)
        self.logger.info("NearestNeighbors index built successfully")

    def recognize_encoding(self, encoding: np.ndarray) -> RecognitionResult:
        if encoding is None or len(encoding) == 0:
            return RecognitionResult(name="Unknown", distance=1.0, confidence=0.0, is_known=False)

        query = np.asarray(encoding, dtype=np.float32).reshape(1, -1)
        distances, indices = self.nn_model.kneighbors(query, n_neighbors=1, return_distance=True)

        distance = float(distances[0][0])
        best_idx = int(indices[0][0])

        is_known = distance < self.threshold
        name = self.known_names[best_idx] if is_known else "Unknown"
        confidence = max(0.0, 1.0 - (distance / self.threshold)) if is_known else 0.0

        return RecognitionResult(name=name, distance=distance, confidence=confidence, is_known=is_known)

    def recognize_batch(self, encodings: Sequence[np.ndarray]) -> List[RecognitionResult]:
        return [self.recognize_encoding(encoding) for encoding in encodings]
