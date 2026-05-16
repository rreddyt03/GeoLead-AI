"""Lightweight tracking primitives for HomeMindAI perception."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

from .detector import DetectionResult


@dataclass
class TrackState:
    """Internal state for a tracked detection."""

    tracking_id: str
    center: tuple[float, float]
    last_seen_at: float
    bounding_box: tuple[int, int, int, int]


class SimpleTracker:
    """Assign temporary IDs using centroid-distance matching.

    This tracker is intentionally lightweight and is designed as a stable
    foundation that can later be replaced by DeepSORT or another advanced
    tracker without changing the pipeline contract.
    """

    def __init__(self, timeout_seconds: float, max_distance: float = 90.0) -> None:
        self._timeout_seconds = timeout_seconds
        self._max_distance = max_distance
        self._tracks: dict[str, TrackState] = {}
        self._next_id = 1

    def update(self, detections: list[DetectionResult]) -> list[DetectionResult]:
        """Assign or reuse tracking IDs for the current detections."""

        now = time.time()
        self._expire_old_tracks(now)

        for detection in detections:
            center = self._center_of(detection.bounding_box)
            best_track_id: str | None = None
            best_distance = float("inf")
            for track_id, track in self._tracks.items():
                distance = math.dist(center, track.center)
                if distance < best_distance and distance <= self._max_distance:
                    best_distance = distance
                    best_track_id = track_id

            if best_track_id is None:
                best_track_id = f"track-{self._next_id}"
                self._next_id += 1

            self._tracks[best_track_id] = TrackState(
                tracking_id=best_track_id,
                center=center,
                last_seen_at=now,
                bounding_box=detection.bounding_box,
            )
            detection.tracking_id = best_track_id

        return detections

    def _expire_old_tracks(self, now: float) -> None:
        """Drop tracks that have not been seen recently."""

        expired_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if now - track.last_seen_at > self._timeout_seconds
        ]
        for track_id in expired_ids:
            del self._tracks[track_id]

    @staticmethod
    def _center_of(bounding_box: tuple[int, int, int, int]) -> tuple[float, float]:
        """Calculate the center point of a bounding box."""

        x1, y1, x2, y2 = bounding_box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)