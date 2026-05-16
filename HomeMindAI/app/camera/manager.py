"""High-level camera orchestration for capture, motion events, and streaming."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time
from typing import Iterator

import numpy as np

from app.events import EventEngine
from app.identity import FaceMatcher, FaceRecognitionEngine, KnownFaceRegistry
from app.utils.image import encode_jpeg, save_frame
from app.vision import FramePipeline, MotionDetector, ProcessedFrame, SimpleTracker, YoloV8PersonDetector
from config import Settings

from .rtsp import RTSPCameraStream
from .stream import (
    CameraSourceType,
    OpenCVCameraStream,
    StreamConfig,
    build_stream_config_from_settings,
)


LOGGER = logging.getLogger(__name__)


class CameraManagerError(RuntimeError):
    """Raised when the camera manager cannot fulfill a request."""


@dataclass
class CameraState:
    """In-memory state for a managed camera."""

    stream: OpenCVCameraStream
    pipeline: FramePipeline
    latest_processed: ProcessedFrame | None = None
    last_motion_snapshot_at: float = 0.0


class CameraManager:
    """Coordinates camera sources, frame processing, and snapshot capture."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._states: dict[str, CameraState] = {}
        self._processing_thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._event_engine = EventEngine()
        self._face_registry: KnownFaceRegistry | None = None

    def start(self) -> None:
        """Start the primary camera stream and processing loop."""

        if self._running:
            return

        stream_config = build_stream_config_from_settings(self._settings)
        stream = self._build_stream(stream_config)
        detector = YoloV8PersonDetector(
            model_path=self._settings.yolo_model_path,
            confidence_threshold=self._settings.yolo_confidence_threshold,
            device=self._settings.yolo_device,
        )
        face_engine = FaceRecognitionEngine(
            model_name=self._settings.insightface_model_name,
            providers=self._settings.insightface_providers,
            detection_size=(self._settings.face_detection_width, self._settings.face_detection_height),
        )
        self._face_registry = KnownFaceRegistry(
            self._settings.photos_dataset_directory,
            embedding_cache_path=self._settings.embedding_cache_path,
        )
        self._face_registry.load(face_engine)
        face_matcher = FaceMatcher(self._settings.face_similarity_threshold)
        tracker = SimpleTracker(timeout_seconds=self._settings.tracking_timeout_seconds)
        motion_detector = MotionDetector(
            area_threshold=self._settings.motion_area_threshold,
            delta_threshold=self._settings.motion_delta_threshold,
            blur_size=self._settings.motion_blur_size,
        )
        pipeline = FramePipeline(
            camera_id=stream.camera_id,
            frame_width=self._settings.frame_width,
            frame_height=self._settings.frame_height,
            motion_detector=motion_detector,
            detector=detector,
            tracker=tracker,
            face_engine=face_engine,
            face_matcher=face_matcher,
            known_face_registry=self._face_registry,
            event_engine=self._event_engine,
            unknown_faces_directory=str(self._settings.unknown_faces_directory),
            inference_stride=self._settings.ai_inference_stride,
        )

        self._states[stream.camera_id] = CameraState(stream=stream, pipeline=pipeline)
        stream.start()

        self._running = True
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            name="homemindai-processing",
            daemon=True,
        )
        self._processing_thread.start()

    def stop(self) -> None:
        """Stop all streams and background workers."""

        self._running = False
        if self._processing_thread is not None:
            self._processing_thread.join(timeout=2)

        for state in self._states.values():
            state.stream.stop()

        self._states.clear()

    def get_camera_ids(self) -> list[str]:
        """Return the active camera identifiers."""

        return list(self._states.keys())

    def list_recent_events(self, limit: int = 25) -> list[dict[str, str | float | None]]:
        """Return recent AI perception events."""

        return self._event_engine.list_recent(limit=limit)

    def reload_known_faces(self) -> dict[str, int]:
        """Reload known face embeddings from disk."""

        if self._face_registry is None:
            raise CameraManagerError("Face registry is not initialized")

        any_state = next(iter(self._states.values()), None)
        if any_state is None or any_state.pipeline._face_engine is None:
            raise CameraManagerError("Face engine is not initialized")

        self._face_registry.load(any_state.pipeline._face_engine)
        return self._face_registry.summary()

    def get_latest_frame(self, camera_id: str | None = None) -> np.ndarray:
        """Return the latest processed frame for a camera."""

        state = self._get_state(camera_id)
        with self._lock:
            if state.latest_processed is None:
                raise CameraManagerError("No frames have been processed yet")
            return state.latest_processed.frame.copy()

    def capture_snapshot(self, camera_id: str | None = None, reason: str = "manual") -> str:
        """Save the latest processed frame to disk and return its path."""

        frame = self.get_latest_frame(camera_id)
        state = self._get_state(camera_id)
        file_path = save_frame(frame, self._settings.snapshots_directory, f"{state.stream.camera_id}_{reason}")
        LOGGER.info(
            "Snapshot saved",
            extra={"camera_id": state.stream.camera_id, "path": str(file_path), "reason": reason},
        )
        return str(file_path)

    def stream_mjpeg(self, camera_id: str | None = None) -> Iterator[bytes]:
        """Yield MJPEG chunks for browser-viewable live streaming."""

        boundary = b"--frame\r\n"
        while True:
            frame = self.get_latest_frame(camera_id)
            jpeg = encode_jpeg(frame)
            yield boundary
            yield b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(1.0 / max(self._settings.fps_limit, 1.0))

    def _processing_loop(self) -> None:
        """Pull frames from each camera stream and run the vision pipeline."""

        while self._running:
            any_frame_processed = False
            for state in self._states.values():
                frame = state.stream.read()
                if frame is None:
                    continue

                processed = state.pipeline.process(frame)
                any_frame_processed = True
                with self._lock:
                    state.latest_processed = processed

                if processed.motion.motion_detected:
                    self._save_motion_snapshot_if_needed(state)

            if not any_frame_processed:
                time.sleep(0.1)

    def _save_motion_snapshot_if_needed(self, state: CameraState) -> None:
        """Rate-limit motion snapshots to avoid excessive writes."""

        now = time.time()
        if now - state.last_motion_snapshot_at < 2.0:
            return

        if state.latest_processed is None:
            return

        file_path = save_frame(
            state.latest_processed.frame,
            self._settings.snapshots_directory,
            f"{state.stream.camera_id}_motion",
        )
        state.last_motion_snapshot_at = now
        LOGGER.info(
            "Motion snapshot saved",
            extra={"camera_id": state.stream.camera_id, "path": str(file_path)},
        )

    def _get_state(self, camera_id: str | None) -> CameraState:
        """Return state for the requested camera, defaulting to the primary camera."""

        if not self._states:
            raise CameraManagerError("Camera manager has not been started")

        resolved_camera_id = camera_id or next(iter(self._states))
        state = self._states.get(resolved_camera_id)
        if state is None:
            raise CameraManagerError(f"Unknown camera id: {resolved_camera_id}")
        return state

    @staticmethod
    def _build_stream(stream_config: StreamConfig) -> OpenCVCameraStream:
        """Instantiate the correct stream adapter for the configured source type."""

        if stream_config.source_type is CameraSourceType.RTSP:
            return RTSPCameraStream(stream_config)
        return OpenCVCameraStream(stream_config)