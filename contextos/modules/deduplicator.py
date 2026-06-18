"""Module 2 -- SemanticDeduplicator.

Removes near-duplicate facts that mean the same thing in different words, using
sentence-transformer embeddings and cosine similarity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class SemanticDeduplicator:
    """Cluster semantically-similar facts and keep one representative per cluster."""

    def __init__(self, model: SentenceTransformer) -> None:
        """Store the embedding model (all-MiniLM-L6-v2).

        Args:
            model: a loaded SentenceTransformer instance.
        """
        # Step 1: Save model on self.
        raise NotImplementedError

    def deduplicate(self, facts: list[str], threshold: float = 0.92) -> list[str]:
        """Return facts with near-duplicates removed.

        Args:
            facts: candidate fact strings.
            threshold: cosine similarity above which two facts count as duplicates.

        Returns:
            A deduplicated list, preserving the longest string per cluster.
        """
        # Step 1: If <= 1 fact, return as-is (nothing to dedupe).
        # Step 2: Embed all facts with self.model.encode(...).
        # Step 3: Compute the pairwise cosine similarity matrix (sklearn).
        # Step 4: Cluster facts connected by similarity >= threshold (union-find / graph).
        # Step 5: For each cluster, keep the longest string (most complete).
        # Step 6: Log facts_before / facts_after counts; return the survivors.
        raise NotImplementedError
