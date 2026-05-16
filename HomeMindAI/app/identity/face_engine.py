"""InsightFace-powered face detection and embedding generation."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


LOGGER = logging.getLogger(__name__)


@dataclass
class FaceDetection:
    """Face detection result with embedding data."""

    bounding_box: tuple[int, int, int, int]
    confidence: float
    embedding: np.ndarray


class FaceRecognitionEngine:
    """Wrapper around InsightFace face analysis and embeddings."""

    def __init__(self, model_name: str, providers: list[str], detection_size: tuple[int, int]) -> None:
        self._analysis = FaceAnalysis(name=model_name, providers=providers)
        self._analysis.prepare(ctx_id=0, det_size=detection_size)
        LOGGER.info(
            "InsightFace engine initialized",
            extra={"model_name": model_name, "providers": providers},
        )

    def detect_faces(self, frame: np.ndarray) -> list[FaceDetection]:
        """Detect faces and return embeddings for the visible frame."""

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self._analysis.get(rgb_frame)
        detections: list[FaceDetection] = []
        for face in faces:
            x1, y1, x2, y2 = [int(value) for value in face.bbox.tolist()]
            detections.append(
                FaceDetection(
                    bounding_box=(x1, y1, x2, y2),
                    confidence=float(face.det_score),
                    embedding=np.asarray(face.embedding, dtype=np.float32),
                )
            )
        return detections

    def load_image(self, image_path: Path) -> np.ndarray | None:
        """Load an image from disk for registry processing."""

        image = cv2.imread(str(image_path))
        if image is None:
            LOGGER.warning("Unable to load face dataset image", extra={"path": str(image_path)})
            return None
        return image

    def extract_largest_face(self, image: np.ndarray) -> FaceDetection | None:
        """Return the largest face found in an image for registry loading."""

        detections = self.detect_faces(image)
        if not detections:
            return None
        return max(detections, key=lambda detection: self._area_of(detection.bounding_box))

    @staticmethod
    def _area_of(bounding_box: tuple[int, int, int, int]) -> int:
        """Return the pixel area of a face box."""

        x1, y1, x2, y2 = bounding_box
        return max(0, x2 - x1) * max(0, y2 - y1)