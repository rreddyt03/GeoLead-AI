"""Face embedding helpers for HomeMindAI identity recognition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FaceEmbedding:
    """Embedding record for a known face sample."""

    identity_key: str
    identity_name: str
    vector: np.ndarray
    source_path: Path


@dataclass
class IdentityEmbeddingProfile:
    """Aggregated embedding profile for a registered identity."""

    identity_key: str
    identity_name: str
    average_vector: np.ndarray
    embeddings: list[FaceEmbedding]
    image_count: int


def normalize_embedding(vector: np.ndarray) -> np.ndarray:
    """Normalize an embedding vector for cosine comparison."""

    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def average_embeddings(embeddings: list[FaceEmbedding]) -> np.ndarray:
    """Average and normalize a list of face embeddings."""

    if not embeddings:
        raise ValueError("Cannot average an empty embedding list")

    stacked = np.stack([normalize_embedding(embedding.vector) for embedding in embeddings])
    return normalize_embedding(np.mean(stacked, axis=0))


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Compute cosine similarity for two embeddings."""

    first_normalized = normalize_embedding(first)
    second_normalized = normalize_embedding(second)
    return float(np.dot(first_normalized, second_normalized))