"""Image utilities for resizing, encoding, and snapshot persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a frame to the configured output size."""

    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def encode_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    """Encode a frame into JPEG bytes for HTTP streaming."""

    success, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise ValueError("Unable to encode frame as JPEG")
    return buffer.tobytes()


def save_frame(frame: np.ndarray, directory: Path, prefix: str) -> Path:
    """Persist a frame to disk using a UTC timestamp-based filename."""

    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    file_path = directory / f"{prefix}_{timestamp}.jpg"

    success = cv2.imwrite(str(file_path), frame)
    if not success:
        raise ValueError(f"Unable to save frame to {file_path}")
    return file_path