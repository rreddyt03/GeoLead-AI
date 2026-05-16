"""Application settings and environment loading for HomeMindAI."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)


def _resolve_existing_path(*candidates: Path) -> Path:
    """Return the first existing path or fall back to the first candidate."""

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


@dataclass
class Settings:
    """Strongly typed application settings."""

    app_name: str
    environment: str
    log_level: str
    camera_source: str
    camera_device_index: int
    rtsp_url: str | None
    frame_width: int
    frame_height: int
    fps_limit: float
    reconnect_interval_seconds: float
    motion_area_threshold: int
    motion_delta_threshold: int
    motion_blur_size: int
    yolo_model_path: str
    yolo_confidence_threshold: float
    yolo_device: str
    insightface_model_name: str
    insightface_providers: list[str]
    face_detection_width: int
    face_detection_height: int
    face_similarity_threshold: float
    tracking_timeout_seconds: float
    ai_inference_stride: int
    photos_dataset_directory: Path
    embedding_cache_path: Path
    log_directory: Path
    known_faces_directory: Path
    unknown_faces_directory: Path
    snapshots_directory: Path
    models_directory: Path
    database_url: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build and cache runtime settings from environment variables."""

    default_photos_directory = _resolve_existing_path(
        BASE_DIR / "photos",
        BASE_DIR / "Photos",
        BASE_DIR / "data" / "known_faces",
    )

    return Settings(
        app_name=os.getenv("HOMEMINDAI_APP_NAME", "HomeMindAI"),
        environment=os.getenv("HOMEMINDAI_ENV", "development"),
        log_level=os.getenv("HOMEMINDAI_LOG_LEVEL", "INFO"),
        camera_source=os.getenv("HOMEMINDAI_CAMERA_SOURCE", "webcam"),
        camera_device_index=int(os.getenv("HOMEMINDAI_CAMERA_DEVICE_INDEX", "0")),
        rtsp_url=os.getenv("HOMEMINDAI_RTSP_URL") or None,
        frame_width=int(os.getenv("HOMEMINDAI_FRAME_WIDTH", "960")),
        frame_height=int(os.getenv("HOMEMINDAI_FRAME_HEIGHT", "540")),
        fps_limit=float(os.getenv("HOMEMINDAI_FPS_LIMIT", "12")),
        reconnect_interval_seconds=float(
            os.getenv("HOMEMINDAI_RECONNECT_INTERVAL_SECONDS", "3")
        ),
        motion_area_threshold=int(os.getenv("HOMEMINDAI_MOTION_AREA_THRESHOLD", "1200")),
        motion_delta_threshold=int(os.getenv("HOMEMINDAI_MOTION_DELTA_THRESHOLD", "25")),
        motion_blur_size=int(os.getenv("HOMEMINDAI_MOTION_BLUR_SIZE", "21")),
        yolo_model_path=os.getenv("HOMEMINDAI_YOLO_MODEL_PATH", "yolov8n.pt"),
        yolo_confidence_threshold=float(os.getenv("HOMEMINDAI_YOLO_CONFIDENCE_THRESHOLD", "0.45")),
        yolo_device=os.getenv("HOMEMINDAI_YOLO_DEVICE", "cpu"),
        insightface_model_name=os.getenv("HOMEMINDAI_INSIGHTFACE_MODEL_NAME", "buffalo_l"),
        insightface_providers=[
            provider.strip()
            for provider in os.getenv(
                "HOMEMINDAI_INSIGHTFACE_PROVIDERS",
                "CPUExecutionProvider",
            ).split(",")
            if provider.strip()
        ],
        face_detection_width=int(os.getenv("HOMEMINDAI_FACE_DETECTION_WIDTH", "640")),
        face_detection_height=int(os.getenv("HOMEMINDAI_FACE_DETECTION_HEIGHT", "640")),
        face_similarity_threshold=float(os.getenv("HOMEMINDAI_FACE_SIMILARITY_THRESHOLD", "0.35")),
        tracking_timeout_seconds=float(os.getenv("HOMEMINDAI_TRACKING_TIMEOUT_SECONDS", "2.5")),
        ai_inference_stride=int(os.getenv("HOMEMINDAI_AI_INFERENCE_STRIDE", "2")),
        photos_dataset_directory=Path(
            os.getenv("HOMEMINDAI_PHOTOS_DATASET_PATH", str(default_photos_directory))
        ),
        embedding_cache_path=Path(
            os.getenv(
                "HOMEMINDAI_EMBEDDING_CACHE_PATH",
                str(BASE_DIR / "data" / "cache" / "family_embeddings.pkl"),
            )
        ),
        log_directory=Path(os.getenv("HOMEMINDAI_LOG_PATH", BASE_DIR / "data" / "logs")),
        known_faces_directory=Path(
            os.getenv("HOMEMINDAI_KNOWN_FACES_PATH", str(default_photos_directory))
        ),
        unknown_faces_directory=Path(
            os.getenv("HOMEMINDAI_UNKNOWN_FACES_PATH", BASE_DIR / "data" / "unknown_faces")
        ),
        snapshots_directory=BASE_DIR / "data" / "snapshots",
        models_directory=BASE_DIR / "models",
        database_url=os.getenv("HOMEMINDAI_DATABASE_URL", "sqlite:///./homemindai.db"),
    )