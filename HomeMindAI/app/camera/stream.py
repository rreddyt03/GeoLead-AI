"""Camera stream abstractions and OpenCV-backed source adapters."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
import threading
import time
from typing import Any

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)


class StreamError(RuntimeError):
    """Raised when a camera stream cannot be opened or maintained."""


class CameraSourceType(str, Enum):
    """Supported camera source types."""

    WEBCAM = "webcam"
    RTSP = "rtsp"


@dataclass
class StreamConfig:
    """Runtime configuration for a single managed camera stream."""

    camera_id: str
    source_type: CameraSourceType
    source: int | str
    frame_width: int
    frame_height: int
    fps_limit: float
    reconnect_interval_seconds: float


class OpenCVCameraStream:
    """Thread-safe OpenCV capture wrapper with reconnect support."""

    def __init__(self, config: StreamConfig) -> None:
        self._config = config
        self._capture: cv2.VideoCapture | None = None
        self._latest_frame: np.ndarray | None = None
        self._latest_timestamp: float = 0.0
        self._running = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    @property
    def camera_id(self) -> str:
        """Return the stable identifier for this camera."""

        return self._config.camera_id

    def start(self) -> None:
        """Start the background capture loop."""

        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, name=f"camera-{self.camera_id}", daemon=True)
        self._thread.start()
        LOGGER.info("Camera stream started", extra={"camera_id": self.camera_id})

    def stop(self) -> None:
        """Stop the background capture loop and release resources."""

        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._release_capture()
        LOGGER.info("Camera stream stopped", extra={"camera_id": self.camera_id})

    def read(self) -> np.ndarray | None:
        """Return the latest frame captured by the background thread."""

        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def last_frame_timestamp(self) -> float:
        """Return the unix timestamp of the latest successful frame."""

        with self._lock:
            return self._latest_timestamp

    def _capture_loop(self) -> None:
        """Continuously pull frames while the stream is running."""

        frame_interval_seconds = 1.0 / self._config.fps_limit if self._config.fps_limit > 0 else 0.0

        while self._running:
            if not self._ensure_capture():
                time.sleep(self._config.reconnect_interval_seconds)
                continue

            assert self._capture is not None
            success, frame = self._capture.read()
            if not success or frame is None:
                LOGGER.warning(
                    "Camera frame read failed; reconnecting",
                    extra={"camera_id": self.camera_id},
                )
                self._release_capture()
                time.sleep(self._config.reconnect_interval_seconds)
                continue

            with self._lock:
                self._latest_frame = frame
                self._latest_timestamp = time.time()

            if frame_interval_seconds > 0:
                time.sleep(frame_interval_seconds)

    def _ensure_capture(self) -> bool:
        """Open the underlying capture device if needed."""

        if self._capture is not None and self._capture.isOpened():
            return True

        self._release_capture()
        capture = cv2.VideoCapture(self._config.source)
        if not capture.isOpened():
            LOGGER.error(
                "Unable to open camera source",
                extra={"camera_id": self.camera_id, "source_type": self._config.source_type.value},
            )
            return False

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._config.frame_width))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._config.frame_height))
        if self._config.fps_limit > 0:
            capture.set(cv2.CAP_PROP_FPS, float(self._config.fps_limit))

        self._capture = capture
        LOGGER.info(
            "Camera source opened",
            extra={"camera_id": self.camera_id, "source_type": self._config.source_type.value},
        )
        return True

    def _release_capture(self) -> None:
        """Release the underlying OpenCV capture object."""

        if self._capture is not None:
            self._capture.release()
            self._capture = None


def build_stream_config_from_settings(settings: Any) -> StreamConfig:
    """Build a primary stream configuration from application settings."""

    source_type = CameraSourceType(settings.camera_source.lower())
    if source_type is CameraSourceType.WEBCAM:
        source: int | str = settings.camera_device_index
        camera_id = f"webcam-{settings.camera_device_index}"
    else:
        if not settings.rtsp_url:
            raise StreamError("RTSP mode requires HOMEMINDAI_RTSP_URL to be configured")
        source = settings.rtsp_url
        camera_id = "rtsp-primary"

    return StreamConfig(
        camera_id=camera_id,
        source_type=source_type,
        source=source,
        frame_width=settings.frame_width,
        frame_height=settings.frame_height,
        fps_limit=settings.fps_limit,
        reconnect_interval_seconds=settings.reconnect_interval_seconds,
    )