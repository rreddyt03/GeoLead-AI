"""Known-face registry loading and caching for HomeMindAI."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import logging
from pathlib import Path
import pickle

from .embeddings import FaceEmbedding, IdentityEmbeddingProfile, average_embeddings, normalize_embedding
from .face_engine import FaceRecognitionEngine


LOGGER = logging.getLogger(__name__)


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_IDENTITY_ALIASES = {
    "rishi": "Rishi",
    "sravan": "Brother",
    "himaja": "Mother",
    "kumar": "Father",
}


@dataclass
class KnownIdentityRecord:
    """Cached embeddings for a single known identity."""

    identity_key: str
    identity_name: str
    embeddings: list[FaceEmbedding] = field(default_factory=list)
    image_count: int = 0
    average_vector: object | None = None

    def to_profile(self) -> IdentityEmbeddingProfile:
        """Convert this record into a matcher-ready profile."""

        if self.average_vector is None:
            raise ValueError(f"Identity {self.identity_name} has no average embedding")
        return IdentityEmbeddingProfile(
            identity_key=self.identity_key,
            identity_name=self.identity_name,
            average_vector=self.average_vector,
            embeddings=self.embeddings,
            image_count=self.image_count,
        )


class KnownFaceRegistry:
    """Loads known family/member faces from the filesystem."""

    def __init__(
        self,
        known_faces_directory: Path,
        embedding_cache_path: Path | None = None,
        identity_aliases: dict[str, str] | None = None,
    ) -> None:
        self._known_faces_directory = known_faces_directory
        self._embedding_cache_path = embedding_cache_path
        self._identity_aliases = identity_aliases or DEFAULT_IDENTITY_ALIASES.copy()
        self._records: dict[str, KnownIdentityRecord] = {}

    def load(self, face_engine: FaceRecognitionEngine) -> None:
        """Load known face images and cache embeddings."""

        self._records.clear()
        self._known_faces_directory.mkdir(parents=True, exist_ok=True)

        dataset_signature = self._build_dataset_signature()
        if self._restore_from_cache(dataset_signature):
            LOGGER.info(
                "Known face registry restored from cache",
                extra={
                    "identity_count": len(self._records),
                    "embedding_count": len(self.all_embeddings()),
                },
            )
            return

        for identity_directory in sorted(self._known_faces_directory.iterdir()):
            if not identity_directory.is_dir():
                continue

            identity_key = self._canonicalize_identity_key(identity_directory.name)
            identity_name = self._identity_aliases.get(identity_key, identity_directory.name.strip().title())
            image_paths = self._collect_identity_images(identity_directory)
            LOGGER.info("Loaded %s images for %s", len(image_paths), identity_name)

            record = KnownIdentityRecord(
                identity_key=identity_key,
                identity_name=identity_name,
                image_count=len(image_paths),
            )
            for image_path in image_paths:
                image = face_engine.load_image(image_path)
                if image is None:
                    continue

                face = face_engine.extract_largest_face(image)
                if face is None:
                    LOGGER.warning("No face found in image %s", image_path.name)
                    continue

                record.embeddings.append(
                    FaceEmbedding(
                        identity_key=identity_key,
                        identity_name=identity_name,
                        vector=normalize_embedding(face.embedding),
                        source_path=image_path,
                    )
                )

            if record.embeddings:
                record.average_vector = average_embeddings(record.embeddings)
                self._records[identity_key] = record
                LOGGER.info("Registered identity: %s", identity_name)
            else:
                LOGGER.warning(
                    "No usable faces were registered for identity",
                    extra={"identity": identity_name, "path": str(identity_directory)},
                )

        self._write_cache(dataset_signature)

        LOGGER.info(
            "Known face registry loaded",
            extra={
                "identity_count": len(self._records),
                "embedding_count": len(self.all_embeddings()),
            },
        )

    def all_embeddings(self) -> list[FaceEmbedding]:
        """Return a flat view of all cached embeddings."""

        embeddings: list[FaceEmbedding] = []
        for record in self._records.values():
            embeddings.extend(record.embeddings)
        return embeddings

    def profiles(self) -> list[IdentityEmbeddingProfile]:
        """Return matcher-ready aggregated identity profiles."""

        return [record.to_profile() for record in self._records.values()]

    def summary(self) -> dict[str, int]:
        """Return a simple summary of the loaded registry."""

        return {record.identity_name: len(record.embeddings) for record in self._records.values()}

    def _collect_identity_images(self, identity_directory: Path) -> list[Path]:
        """Recursively collect all supported image files for one identity."""

        return sorted(
            path
            for path in identity_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )

    def _build_dataset_signature(self) -> str:
        """Build a stable signature for cache invalidation."""

        digest = hashlib.sha256()
        for path in sorted(self._known_faces_directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue
            stat = path.stat()
            digest.update(str(path.relative_to(self._known_faces_directory)).encode("utf-8", errors="ignore"))
            digest.update(str(stat.st_mtime_ns).encode("utf-8"))
            digest.update(str(stat.st_size).encode("utf-8"))
        return digest.hexdigest()

    def _restore_from_cache(self, dataset_signature: str) -> bool:
        """Load cached embeddings when the dataset signature matches."""

        if self._embedding_cache_path is None or not self._embedding_cache_path.exists():
            return False

        try:
            with self._embedding_cache_path.open("rb") as cache_file:
                payload = pickle.load(cache_file)
        except Exception as exc:
            LOGGER.warning("Failed to read embedding cache", extra={"error": str(exc)})
            return False

        if payload.get("dataset_signature") != dataset_signature:
            return False

        restored_records: dict[str, KnownIdentityRecord] = {}
        for item in payload.get("records", []):
            embeddings = [
                FaceEmbedding(
                    identity_key=item["identity_key"],
                    identity_name=item["identity_name"],
                    vector=embedding["vector"],
                    source_path=Path(embedding["source_path"]),
                )
                for embedding in item["embeddings"]
            ]
            restored_records[item["identity_key"]] = KnownIdentityRecord(
                identity_key=item["identity_key"],
                identity_name=item["identity_name"],
                embeddings=embeddings,
                image_count=item["image_count"],
                average_vector=item["average_vector"],
            )

        self._records = restored_records
        return True

    def _write_cache(self, dataset_signature: str) -> None:
        """Persist cached embeddings for faster future startups."""

        if self._embedding_cache_path is None:
            return

        self._embedding_cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset_signature": dataset_signature,
            "records": [
                {
                    "identity_key": record.identity_key,
                    "identity_name": record.identity_name,
                    "image_count": record.image_count,
                    "average_vector": record.average_vector,
                    "embeddings": [
                        {
                            "source_path": str(embedding.source_path),
                            "vector": embedding.vector,
                        }
                        for embedding in record.embeddings
                    ],
                }
                for record in self._records.values()
            ],
        }
        try:
            with self._embedding_cache_path.open("wb") as cache_file:
                pickle.dump(payload, cache_file)
        except Exception as exc:
            LOGGER.warning("Failed to write embedding cache", extra={"error": str(exc)})

    @staticmethod
    def _canonicalize_identity_key(folder_name: str) -> str:
        """Normalize a folder name into a stable identity key."""

        return folder_name.strip().lower()