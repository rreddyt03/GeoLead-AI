"""Identity and face recognition workflows."""

from .embeddings import FaceEmbedding, IdentityEmbeddingProfile, average_embeddings, cosine_similarity, normalize_embedding
from .face_engine import FaceDetection, FaceRecognitionEngine
from .matcher import FaceMatcher, IdentityMatch
from .registry import KnownFaceRegistry, KnownIdentityRecord

__all__ = [
	"FaceDetection",
	"FaceEmbedding",
	"FaceMatcher",
	"FaceRecognitionEngine",
	"IdentityEmbeddingProfile",
	"IdentityMatch",
	"KnownFaceRegistry",
	"KnownIdentityRecord",
	"average_embeddings",
	"cosine_similarity",
	"normalize_embedding",
]