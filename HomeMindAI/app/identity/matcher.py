"""Identity matching utilities for HomeMindAI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embeddings import IdentityEmbeddingProfile, cosine_similarity


@dataclass
class IdentityMatch:
    """Result of matching a face embedding against the known registry."""

    identity: str
    similarity: float
    is_known: bool
    source_path: str | None = None
    matched_by: str | None = None


class FaceMatcher:
    """Matches live face embeddings against stored family embeddings."""

    def __init__(self, similarity_threshold: float) -> None:
        self._similarity_threshold = similarity_threshold

    def match(self, embedding: np.ndarray, profiles: list[IdentityEmbeddingProfile]) -> IdentityMatch:
        """Return the closest known identity or UNKNOWN."""

        best_match = IdentityMatch(identity="UNKNOWN", similarity=0.0, is_known=False)
        for profile in profiles:
            average_similarity = cosine_similarity(embedding, profile.average_vector)
            best_sample_score = 0.0
            best_sample_path: str | None = None
            for sample in profile.embeddings:
                sample_similarity = cosine_similarity(embedding, sample.vector)
                if sample_similarity > best_sample_score:
                    best_sample_score = sample_similarity
                    best_sample_path = str(sample.source_path)

            similarity = max(average_similarity, best_sample_score)
            matched_by = "average_embedding" if average_similarity >= best_sample_score else "best_sample"
            if similarity > best_match.similarity:
                best_match = IdentityMatch(
                    identity=profile.identity_name,
                    similarity=similarity,
                    is_known=similarity >= self._similarity_threshold,
                    source_path=best_sample_path,
                    matched_by=matched_by,
                )

        if not best_match.is_known:
            return IdentityMatch(identity="UNKNOWN", similarity=best_match.similarity, is_known=False)
        return best_match