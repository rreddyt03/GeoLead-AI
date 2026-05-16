"""Reusable frame-processing pipeline for HomeMindAI camera streams."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import time

import numpy as np

from app.events import EventEngine, EventType, PerceptionEvent
from app.identity import FaceMatcher, FaceRecognitionEngine, KnownFaceRegistry
from app.utils.drawing import draw_detections, draw_faces, draw_motion_status
from app.utils.image import resize_frame, save_frame
from .detector import DetectionResult, DetectorBase, NoOpDetector
from .motion import MotionDetectionResult, MotionDetector
from .tracker import SimpleTracker


LOGGER = logging.getLogger(__name__)


@dataclass
class ProcessedFrame:
    """Pipeline output for a single frame."""

    frame: np.ndarray
    motion: MotionDetectionResult
    detections: list[DetectionResult] = field(default_factory=list)
    events: list[PerceptionEvent] = field(default_factory=list)


class FramePipeline:
    """Composes frame resizing, motion analysis, and future AI inference."""

    def __init__(
        self,
        camera_id: str,
        frame_width: int,
        frame_height: int,
        motion_detector: MotionDetector,
        detector: DetectorBase | None = None,
        tracker: SimpleTracker | None = None,
        face_engine: FaceRecognitionEngine | None = None,
        face_matcher: FaceMatcher | None = None,
        known_face_registry: KnownFaceRegistry | None = None,
        event_engine: EventEngine | None = None,
        unknown_faces_directory: str | None = None,
        inference_stride: int = 1,
    ) -> None:
        self._camera_id = camera_id
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._motion_detector = motion_detector
        self._detector = detector or NoOpDetector()
        self._tracker = tracker
        self._face_engine = face_engine
        self._face_matcher = face_matcher
        self._known_face_registry = known_face_registry
        self._event_engine = event_engine
        self._unknown_faces_directory = unknown_faces_directory
        self._inference_stride = max(inference_stride, 1)
        self._frame_counter = 0
        self._last_detections: list[DetectionResult] = []
        self._last_faces = []
        self._unknown_track_ids: set[str] = set()
        self._last_tracked_event_at: dict[str, float] = {}
        self._last_identity_event: dict[str, tuple[str, float]] = {}

    def process(self, frame: np.ndarray) -> ProcessedFrame:
        """Run the configured processing stages against a frame."""

        resized_frame = resize_frame(frame, self._frame_width, self._frame_height)
        motion_result = self._motion_detector.analyze(resized_frame)
        self._frame_counter += 1

        run_ai_inference = (self._frame_counter % self._inference_stride) == 0
        detections = self._last_detections
        faces = self._last_faces
        events: list[PerceptionEvent] = []

        if run_ai_inference:
            detections = self._detector.detect(resized_frame)
            if self._tracker is not None:
                detections = self._tracker.update(detections)

            if self._face_engine is not None:
                faces = self._face_engine.detect_faces(resized_frame)
                events = self._apply_identity_and_events(resized_frame, detections, faces)

            self._last_detections = [self._copy_detection(detection) for detection in detections]
            self._last_faces = faces

        annotated_frame = resized_frame.copy()
        draw_motion_status(annotated_frame, motion_result)
        draw_detections(annotated_frame, detections)
        draw_faces(annotated_frame, faces)

        if motion_result.motion_detected:
            LOGGER.info(
                "Motion detected",
                extra={
                    "contour_count": motion_result.contour_count,
                    "motion_score": motion_result.motion_score,
                },
            )

        return ProcessedFrame(
            frame=annotated_frame,
            motion=motion_result,
            detections=[self._copy_detection(detection) for detection in detections],
            events=events,
        )

    def _apply_identity_and_events(
        self,
        frame: np.ndarray,
        detections: list[DetectionResult],
        faces: list,
    ) -> list[PerceptionEvent]:
        """Attach identities, save unknown visitors, and generate events."""

        if self._face_matcher is None or self._known_face_registry is None or self._event_engine is None:
            return []

        events: list[PerceptionEvent] = []
        identity_profiles = self._known_face_registry.profiles()

        for detection in detections:
            self._emit_tracked_event_if_needed(detection, events)
            face = self._find_face_for_person(detection.bounding_box, faces)
            if face is None:
                detection.identity = "FACE_NOT_VISIBLE"
                continue

            match = self._face_matcher.match(face.embedding, identity_profiles)
            detection.identity = match.identity
            detection.identity_confidence = match.similarity
            if not self._should_emit_identity_event(detection, match.identity):
                continue

            if match.is_known:
                events.append(
                    self._event_engine.emit(
                        event_type=EventType.KNOWN_PERSON_DETECTED,
                        camera_id=self._camera_id,
                        identity=match.identity,
                        confidence=match.similarity,
                        snapshot_path=None,
                        tracking_id=detection.tracking_id,
                    )
                )
            else:
                snapshot_path = self._save_unknown_face(frame, face.bounding_box, detection.tracking_id)
                events.append(
                    self._event_engine.emit(
                        event_type=EventType.UNKNOWN_PERSON_DETECTED,
                        camera_id=self._camera_id,
                        identity="UNKNOWN",
                        confidence=match.similarity,
                        snapshot_path=snapshot_path,
                        tracking_id=detection.tracking_id,
                    )
                )

        return events

    def _should_emit_identity_event(self, detection: DetectionResult, identity: str) -> bool:
        """Emit identity events only when a track changes identity or enough time has passed."""

        if detection.tracking_id is None:
            return True

        now = time.time()
        previous = self._last_identity_event.get(detection.tracking_id)
        if previous is not None:
            previous_identity, previous_timestamp = previous
            if previous_identity == identity and now - previous_timestamp < 2.5:
                return False

        self._last_identity_event[detection.tracking_id] = (identity, now)
        return True

    def _emit_tracked_event_if_needed(
        self,
        detection: DetectionResult,
        events: list[PerceptionEvent],
    ) -> None:
        """Rate-limit generic tracking events to keep event volume manageable."""

        if detection.tracking_id is None or self._event_engine is None:
            return

        now = time.time()
        last_seen = self._last_tracked_event_at.get(detection.tracking_id, 0.0)
        if now - last_seen < 1.0:
            return

        self._last_tracked_event_at[detection.tracking_id] = now
        events.append(
            self._event_engine.emit(
                event_type=EventType.PERSON_TRACKED,
                camera_id=self._camera_id,
                identity=detection.identity or "UNIDENTIFIED",
                confidence=detection.confidence,
                snapshot_path=None,
                tracking_id=detection.tracking_id,
            )
        )

    def _save_unknown_face(
        self,
        frame: np.ndarray,
        bounding_box: tuple[int, int, int, int],
        tracking_id: str | None,
    ) -> str | None:
        """Persist a cropped face image for previously unseen visitors."""

        if self._unknown_faces_directory is None:
            return None
        if tracking_id is not None and tracking_id in self._unknown_track_ids:
            return None

        x1, y1, x2, y2 = bounding_box
        face_crop = frame[max(0, y1):max(y1, y2), max(0, x1):max(x1, x2)]
        if face_crop.size == 0:
            return None

        snapshot_path = save_frame(
            face_crop,
            Path(self._unknown_faces_directory),
            f"unknown_{tracking_id or 'face'}",
        )
        if tracking_id is not None:
            self._unknown_track_ids.add(tracking_id)
        return str(snapshot_path)

    @staticmethod
    def _find_face_for_person(person_box: tuple[int, int, int, int], faces: list) -> object | None:
        """Find the face whose center lies inside a person box."""

        x1, y1, x2, y2 = person_box
        for face in faces:
            fx1, fy1, fx2, fy2 = face.bounding_box
            center_x = (fx1 + fx2) / 2.0
            center_y = (fy1 + fy2) / 2.0
            if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                return face
        return None

    @staticmethod
    def _copy_detection(detection: DetectionResult) -> DetectionResult:
        """Return a detached copy of a detection result."""

        return DetectionResult(
            label=detection.label,
            confidence=detection.confidence,
            bounding_box=detection.bounding_box,
            tracking_id=detection.tracking_id,
            identity=detection.identity,
            identity_confidence=detection.identity_confidence,
            metadata=dict(detection.metadata),
        )