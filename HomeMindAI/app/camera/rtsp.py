"""RTSP-specific stream adapter for HomeMindAI."""

from __future__ import annotations

from .stream import OpenCVCameraStream, StreamConfig


class RTSPCameraStream(OpenCVCameraStream):
    """Dedicated RTSP stream type for clearer extension points.

    This class currently inherits the OpenCV behavior directly, but keeping it
    separate makes it straightforward to add RTSP-specific buffering, transport
    tuning, or connection diagnostics later.
    """

    def __init__(self, config: StreamConfig) -> None:
        super().__init__(config)