from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import face_recognition
import numpy as np
from tqdm import tqdm


@dataclass
class EncodingSummary:
    total_images: int
    processed_images: int
    skipped_images: int
    students_found: int
    encodings_saved: int


class FaceEncoder:
    """Build and persist face encodings from a student image dataset."""

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, model: str = "hog", num_jitters: int = 1) -> None:
        self.model = model
        self.num_jitters = max(1, int(num_jitters))
        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_image_paths(self, dataset_dir: Path) -> List[Tuple[str, Path]]:
        image_paths: List[Tuple[str, Path]] = []

        for student_dir in sorted(dataset_dir.iterdir()):
            if not student_dir.is_dir():
                continue

            student_name = student_dir.name
            for image_path in sorted(student_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in self.VALID_EXTENSIONS:
                    image_paths.append((student_name, image_path))

        return image_paths

    @staticmethod
    def _largest_face(face_locations: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
        return max(face_locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))

    def build_encodings(self, dataset_dir: Path) -> Tuple[Dict[str, object], EncodingSummary]:
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

        image_paths = self._get_image_paths(dataset_dir)
        if not image_paths:
            raise ValueError(f"No valid images found in dataset directory: {dataset_dir}")

        known_encodings: List[np.ndarray] = []
        known_names: List[str] = []
        skipped_images = 0

        for student_name, image_path in tqdm(image_paths, desc="Encoding faces", unit="image"):
            try:
                image = face_recognition.load_image_file(str(image_path))
                face_locations = face_recognition.face_locations(image, model=self.model)

                if not face_locations:
                    skipped_images += 1
                    self.logger.warning("No face found in image: %s", image_path)
                    continue

                target_location = self._largest_face(face_locations)
                face_encoding = face_recognition.face_encodings(
                    image,
                    known_face_locations=[target_location],
                    num_jitters=self.num_jitters,
                )

                if not face_encoding:
                    skipped_images += 1
                    self.logger.warning("Face encoding failed for image: %s", image_path)
                    continue

                known_encodings.append(face_encoding[0])
                known_names.append(student_name)
            except Exception as exc:  # pragma: no cover - defensive path
                skipped_images += 1
                self.logger.exception("Failed to process image '%s': %s", image_path, exc)

        if not known_encodings:
            raise RuntimeError("No face encodings generated. Check dataset quality and face visibility.")

        encodings_array = np.asarray(known_encodings, dtype=np.float32)
        payload: Dict[str, object] = {
            "encodings": encodings_array,
            "names": known_names,
            "metadata": {
                "model": self.model,
                "num_jitters": self.num_jitters,
                "students": sorted(set(known_names)),
                "total_samples": int(encodings_array.shape[0]),
                "embedding_size": int(encodings_array.shape[1]),
            },
        }

        summary = EncodingSummary(
            total_images=len(image_paths),
            processed_images=len(known_encodings),
            skipped_images=skipped_images,
            students_found=len(set(known_names)),
            encodings_saved=len(known_encodings),
        )

        return payload, summary

    def save_encodings(self, payload: Dict[str, object], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as file:
            pickle.dump(payload, file)

    def train(self, dataset_dir: Path, output_path: Path) -> EncodingSummary:
        payload, summary = self.build_encodings(dataset_dir)
        self.save_encodings(payload, output_path)
        self.logger.info("Encodings written to: %s", output_path)
        return summary
