"""Application entry point for HomeMindAI AI perception stack."""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse

from app.camera import CameraConfig, CameraManager, CameraManagerError, CameraService, CameraServiceError
from app.utils import configure_logging
from config import get_settings


LOGGER = logging.getLogger(__name__)
CAMERA_MANAGER: CameraManager | None = None


def ensure_directories(paths: list[Path]) -> None:
    """Create runtime directories required by the application."""

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def bootstrap_runtime() -> None:
    """Initialize directories and logging for the application runtime."""

    settings = get_settings()
    ensure_directories(
        [
            settings.log_directory,
            settings.known_faces_directory,
            settings.unknown_faces_directory,
            settings.snapshots_directory,
            settings.models_directory,
        ]
    )
    configure_logging(log_directory=settings.log_directory, level=settings.log_level)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Manage long-lived services for FastAPI startup and shutdown."""

    del application
    global CAMERA_MANAGER

    bootstrap_runtime()
    settings = get_settings()
    CAMERA_MANAGER = CameraManager(settings)
    try:
        CAMERA_MANAGER.start()
        yield
    finally:
        if CAMERA_MANAGER is not None:
            CAMERA_MANAGER.stop()
            CAMERA_MANAGER = None


def build_application() -> FastAPI:
    """Create the FastAPI application for future API expansion."""

    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="AI-powered home intelligence surveillance system foundation.",
        lifespan=lifespan,
    )

    @app.get("/health", tags=["system"])
    def health_check() -> dict[str, str | list[str] | int]:
        """Return a simple health signal for local development."""

        camera_ids = CAMERA_MANAGER.get_camera_ids() if CAMERA_MANAGER is not None else []
        return {
            "status": "ok",
            "service": settings.app_name,
            "environment": settings.environment,
            "cameras": camera_ids,
            "inference_stride": settings.ai_inference_stride,
        }

    @app.get("/camera/live", tags=["camera"])
    def live_stream(camera_id: str | None = Query(default=None)) -> StreamingResponse:
        """Serve a browser-viewable MJPEG stream for the requested camera."""

        if CAMERA_MANAGER is None:
            raise HTTPException(status_code=503, detail="Camera manager is not running")

        try:
            stream = CAMERA_MANAGER.stream_mjpeg(camera_id=camera_id)
        except CameraManagerError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        return StreamingResponse(
            stream,
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.post("/camera/snapshot", tags=["camera"])
    def capture_snapshot(camera_id: str | None = Query(default=None)) -> JSONResponse:
        """Persist a snapshot of the latest processed frame."""

        if CAMERA_MANAGER is None:
            raise HTTPException(status_code=503, detail="Camera manager is not running")

        try:
            path = CAMERA_MANAGER.capture_snapshot(camera_id=camera_id, reason="manual")
        except CameraManagerError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        return JSONResponse(content={"snapshot_path": path})

    @app.get("/events/recent", tags=["events"])
    def list_recent_events(limit: int = Query(default=25, ge=1, le=100)) -> JSONResponse:
        """Return recent perception and identity events."""

        if CAMERA_MANAGER is None:
            raise HTTPException(status_code=503, detail="Camera manager is not running")

        return JSONResponse(content={"events": CAMERA_MANAGER.list_recent_events(limit=limit)})

    @app.post("/identity/reload-known-faces", tags=["identity"])
    def reload_known_faces() -> JSONResponse:
        """Reload known face embeddings from the configured directory."""

        if CAMERA_MANAGER is None:
            raise HTTPException(status_code=503, detail="Camera manager is not running")

        try:
            summary = CAMERA_MANAGER.reload_known_faces()
        except CameraManagerError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return JSONResponse(content={"known_faces": summary})

    return app


def run_camera_test() -> int:
    """Run the HomeMindAI webcam verification flow.

    Returns:
        A process exit code suitable for shell usage.
    """

    settings = get_settings()
    bootstrap_runtime()

    LOGGER.info("HomeMindAI foundation bootstrapped")
    service = CameraService(
        CameraConfig(
            device_index=settings.camera_device_index,
            window_title="HomeMindAI Camera Test",
            snapshot_directory=settings.snapshots_directory,
        )
    )

    try:
        service.run_camera_test(frame_width=settings.frame_width, frame_height=settings.frame_height)
    except CameraServiceError as exc:
        LOGGER.exception("Camera test failed: %s", exc)
        return 1

    LOGGER.info("Camera test completed successfully")
    return 0


app = build_application()


if __name__ == "__main__":
    sys.exit(run_camera_test())