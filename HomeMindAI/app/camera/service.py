"""Camera service primitives for local capture and future stream adapters."""

from __future__ import annotations

from dataclasses import dataclass
import logging

from pathlib import Path

import cv2


LOGGER = logging.getLogger(__name__)


class CameraServiceError(RuntimeError):
    """Raised when the camera service cannot start or read frames."""


@dataclass
class CameraConfig:
    """Runtime settings for a camera capture session."""

    device_index: int = 0
    window_title: str = "HomeMindAI Camera Test"
    exit_key: str = "q"
    snapshot_directory: Path | None = None


class CameraService:
    """Small service wrapper around OpenCV camera capture.

    This class intentionally keeps the surface area small so future
    implementations can swap the webcam source for RTSP, NVR, or managed
    edge devices without changing the application entry point.
    """

    def __init__(self, config: CameraConfig) -> None:
        self._config = config

    def run_camera_test(self, frame_width: int | None = None, frame_height: int | None = None) -> None:
        """Run a local camera test until the operator presses Q."""

        LOGGER.info(
            "Starting camera test",
            extra={"device_index": self._config.device_index},
        )
        capture = cv2.VideoCapture(self._config.device_index)

        if frame_width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(frame_width))
        if frame_height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(frame_height))

        if not capture.isOpened():
            LOGGER.error("Unable to open webcam", extra={"device_index": self._config.device_index})
            raise CameraServiceError(
                "Unable to access the default webcam. "
                "Check camera permissions and verify that no other app is using it."
            )

        try:
            while True:
                success, frame = capture.read()
                if not success:
                    LOGGER.warning("Frame capture failed; stopping camera test")
                    raise CameraServiceError(
                        "The webcam opened, but HomeMindAI could not read frames from it."
                    )

                # Show the raw test stream so the operator can verify the device.
                cv2.imshow(self._config.window_title, frame)

                pressed_key = cv2.waitKey(1) & 0xFF
                if pressed_key == ord(self._config.exit_key.lower()):
                    LOGGER.info("Camera test stopped by operator")
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()
            LOGGER.info("Camera resources released")