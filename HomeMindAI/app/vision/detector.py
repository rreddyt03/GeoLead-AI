"""Detection abstractions for YOLO and other inference backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DetectionResult:
    """Normalized detection output for downstream consumers."""

    label: str
    confidence: float
    bounding_box: tuple[int, int, int, int]
    tracking_id: str | None = None
    identity: str | None = None
    identity_confidence: float | None = None
    metadata: dict[str, str | float | int] = field(default_factory=dict)


class DetectorBase(ABC):
    """Contract for future inference engines.

    Concrete implementations may wrap YOLO, InsightFace, or other custom
    computer-vision models while preserving a stable output format.
    """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        """Run inference against a single frame."""


class NoOpDetector(DetectorBase):
    """Placeholder detector used until real AI models are enabled."""

    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        """Return no detections while preserving the detector contract."""

        _ = frame
        return []