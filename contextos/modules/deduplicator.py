"""Module 2 -- SemanticDeduplicator.

Removes near-duplicate facts that mean the same thing in different words, using
sentence-transformer embeddings and cosine similarity.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


class SemanticDeduplicator:
    """Cluster semantically-similar facts and keep one representative per cluster."""

    def __init__(self, model: SentenceTransformer) -> None:
        """Store the embedding model (all-MiniLM-L6-v2).

        Args:
            model: a loaded SentenceTransformer instance.
        """
        # Step 1: Save model on self.
        self.model = model

    def deduplicate(self, facts: list[str], threshold: float = 0.92) -> list[str]:
        """Return facts with near-duplicates removed.

        Args:
            facts: candidate fact strings.
            threshold: cosine similarity above which two facts count as duplicates.

        Returns:
            A deduplicated list, preserving the longest string per cluster.
        """
        # Step 1: If <= 1 fact, return as-is (nothing to dedupe).
        if len(facts) <= 1:
            return facts

        # Step 2: Embed all facts with self.model.encode(...).
        embeddings = self.model.encode(facts)

        # Step 3: Compute the pairwise cosine similarity matrix (sklearn).
        sim = cosine_similarity(embeddings)

        # Step 4: Cluster facts connected by similarity >= threshold (union-find).
        # Every fact starts in its own cluster; each similar pair merges clusters,
        # so transitively-similar facts (A~B, B~C) end up in one cluster.
        n = len(facts)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            parent[find(a)] = find(b)

        for i in range(n):
            for j in range(i + 1, n):
                if sim[i][j] >= threshold:
                    union(i, j)

        clusters: dict[int, list[int]] = {}
        for i in range(n):
            clusters.setdefault(find(i), []).append(i)

        # Step 5: For each cluster, keep the longest string (most complete).
        survivors = [
            max(members, key=lambda i: len(facts[i])) for members in clusters.values()
        ]
        # Preserve the original input order of the survivors.
        result = [facts[i] for i in sorted(survivors)]

        # Step 6: Log facts_before / facts_after counts; return the survivors.
        logger.info("facts_before=%d facts_after=%d", len(facts), len(result))
        return result
