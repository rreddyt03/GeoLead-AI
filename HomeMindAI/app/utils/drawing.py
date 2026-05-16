"""Drawing helpers for HomeMindAI overlays."""

from __future__ import annotations

import cv2
import numpy as np

from app.identity.face_engine import FaceDetection
from app.vision.detector import DetectionResult
from app.vision.motion import MotionDetectionResult


def draw_motion_status(frame: np.ndarray, motion_result: MotionDetectionResult) -> None:
    """Overlay motion state and contour rectangles."""

    status = "MOTION" if motion_result.motion_detected else "STABLE"
    status_color = (0, 0, 255) if motion_result.motion_detected else (0, 180, 0)
    cv2.putText(frame, f"Status: {status}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    for x, y, width, height in motion_result.bounding_boxes:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)


def draw_detections(frame: np.ndarray, detections: list[DetectionResult]) -> None:
    """Overlay person detections, tracking IDs, and identity labels."""

    for detection in detections:
        x1, y1, x2, y2 = detection.bounding_box
        color = (40, 180, 255)
        if detection.identity == "UNKNOWN":
            color = (0, 0, 255)
        elif detection.identity and detection.identity not in {"UNKNOWN", "FACE_NOT_VISIBLE"}:
            color = (0, 200, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label_parts: list[str] = []
        if detection.identity:
            confidence = detection.identity_confidence if detection.identity_confidence is not None else detection.confidence
            label_parts.append(f"{detection.identity} | {confidence * 100:.0f}%")
        else:
            label_parts.append(f"{detection.label} | {detection.confidence * 100:.0f}%")
        if detection.tracking_id:
            label_parts.append(detection.tracking_id)

        cv2.putText(
            frame,
            " | ".join(label_parts),
            (x1, max(y1 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )


def draw_faces(frame: np.ndarray, faces: list[FaceDetection]) -> None:
    """Overlay face boxes for debugging recognition behavior."""

    for face in faces:
        x1, y1, x2, y2 = face.bounding_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 120, 0), 1)