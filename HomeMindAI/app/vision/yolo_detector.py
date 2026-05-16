"""YOLOv8-based person detector for HomeMindAI."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from .detector import DetectionResult, DetectorBase


LOGGER = logging.getLogger(__name__)


class YoloV8PersonDetector(DetectorBase):
    """Ultralytics YOLOv8 detector limited to person class inference."""

    PERSON_CLASS_ID = 0

    def __init__(self, model_path: str, confidence_threshold: float, device: str = "cpu") -> None:
        self._model_path = model_path
        self._confidence_threshold = confidence_threshold
        self._device = device
        self._model = YOLO(model_path)
        LOGGER.info(
            "YOLO detector initialized",
            extra={"model_path": model_path, "device": device},
        )

    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        """Run person-only detection for a single frame."""

        results = self._model.predict(
            source=frame,
            verbose=False,
            conf=self._confidence_threshold,
            classes=[self.PERSON_CLASS_ID],
            device=self._device,
        )

        detections: list[DetectionResult] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                detections.append(
                    DetectionResult(
                        label="person",
                        confidence=confidence,
                        bounding_box=(x1, y1, x2, y2),
                        metadata={
                            "class_id": self.PERSON_CLASS_ID,
                            "model_path": str(self._model_path),
                        },
                    )
                )

        return detections