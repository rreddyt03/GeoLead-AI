"""Lightweight motion detection utilities built on top of OpenCV."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MotionDetectionResult:
    """Output of the motion detector for a single frame."""

    motion_detected: bool
    contour_count: int
    bounding_boxes: list[tuple[int, int, int, int]]
    motion_score: float
    debug_mask: np.ndarray | None = None


class MotionDetector:
    """Simple frame-differencing motion detector.

    The detector keeps the previous blurred grayscale frame in memory and uses
    frame differencing, thresholding, and contour extraction to flag motion.
    """

    def __init__(
        self,
        area_threshold: int = 1200,
        delta_threshold: int = 25,
        blur_size: int = 21,
    ) -> None:
        self._area_threshold = area_threshold
        self._delta_threshold = delta_threshold
        self._blur_size = blur_size
        self._previous_frame: np.ndarray | None = None

    def analyze(self, frame: np.ndarray) -> MotionDetectionResult:
        """Analyze a frame and return motion metadata."""

        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grayscale, (self._blur_size, self._blur_size), 0)

        if self._previous_frame is None:
            self._previous_frame = blurred
            return MotionDetectionResult(False, 0, [], 0.0)

        frame_delta = cv2.absdiff(self._previous_frame, blurred)
        _, threshold_mask = cv2.threshold(
            frame_delta,
            self._delta_threshold,
            255,
            cv2.THRESH_BINARY,
        )
        dilated_mask = cv2.dilate(threshold_mask, None, iterations=2)
        contours, _ = cv2.findContours(
            dilated_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        bounding_boxes: list[tuple[int, int, int, int]] = []
        motion_score = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self._area_threshold:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, width, height))
            motion_score += float(area)

        self._previous_frame = blurred
        return MotionDetectionResult(
            motion_detected=bool(bounding_boxes),
            contour_count=len(bounding_boxes),
            bounding_boxes=bounding_boxes,
            motion_score=motion_score,
            debug_mask=dilated_mask,
        )