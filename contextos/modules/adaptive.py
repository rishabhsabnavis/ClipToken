"""Module 4 -- AdaptiveCompressor (the differentiator).

Replaces the old fixed-tier "summarization pyramid." Instead of compressing turns by
age, it compresses each turn by its *predicted impact on the agent's next action*,
choosing the most aggressive level whose predicted impact stays under a budget.

Before compressing anything it writes the original to the FidelityStore, so every
decision is reversible -- this is what lets loss fall toward zero over time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anthropic import Anthropic

    from contextos.learning.policy import CompressionPolicy
    from contextos.learning.store import FidelityStore
    from contextos.schemas.models import Segment, Turn


class AdaptiveCompressor:
    """Decide and apply a per-turn compression level using the learned policy."""

    def __init__(
        self,
        client: Anthropic,
        model: str,
        store: FidelityStore,
        coldstart_tier1: int = 5,
        coldstart_tier2: int = 10,
    ) -> None:
        """Wire up dependencies and the cold-start age prior.

        Args:
            client: Anthropic client for Haiku summaries.
            model: Haiku model id.
            store: FidelityStore, written to before any lossy compression.
            coldstart_tier1: age boundary for the cold-start fallback (-> sentence).
            coldstart_tier2: age boundary for the cold-start fallback (-> bullet).
        """
        # Step 1: Save client, model, store, and the two cold-start boundaries.
        raise NotImplementedError

    def compress_history(self, turns: list[Turn], policy: CompressionPolicy) -> list[Turn]:
        """Compress each turn to the level the policy chooses (or the cold-start prior).

        Args:
            turns: full session turn history (oldest first).
            policy: the CompressionPolicy that maps a Segment -> level.

        Returns:
            The same turns with `level`, `summary`, and `fidelity_ref` populated.
        """
        # Step 1: For each turn, build a Segment (self._featurize) describing it.
        # Step 2: Ask policy.decide(segment) for a level; if the policy has no data yet
        #         fall back to self._coldstart_level(segment).
        # Step 3: If the level is lossy (not "verbatim"), store the original in the
        #         FidelityStore (store.put) and record the returned hash on turn.fidelity_ref.
        # Step 4: Apply the level via self._apply_level(turn, level).
        # Step 5: Log per-turn level + predicted impact + tokens before/after.
        # Step 6: Return the updated turns.
        raise NotImplementedError

    def _featurize(self, turn: Turn, turns: list[Turn]) -> Segment:
        """Build the feature vector the policy needs for one turn.

        Args:
            turn: the turn being scored.
            turns: full history (for age, references, relevance to the latest goal).

        Returns:
            A Segment describing this turn.
        """
        # Step 1: age_turns = len(turns) - 1 - turn.index.
        # Step 2: token_len = token count of turn.content.
        # Step 3: tool_name / segment_type = infer from the turn's content shape.
        # Step 4: semantic_relevance = cosine sim of this turn to the latest user goal.
        # Step 5: times_referenced_recently = how often recent turns mention this content.
        # Step 6: Assemble and return the Segment.
        raise NotImplementedError

    def _coldstart_level(self, segment: Segment) -> str:
        """Age-based fallback level, used before the policy has learned anything.

        Args:
            segment: the segment being scored.

        Returns:
            "sentence" | "bullet" | "verbatim" per the cold-start tiers.
        """
        # Step 1: age <= coldstart_tier1 ... (oldest) -> "sentence".
        # Step 2: age <= coldstart_tier2 -> "bullet".
        # Step 3: else -> "verbatim".
        raise NotImplementedError

    def _apply_level(self, turn: Turn, level: str) -> Turn:
        """Apply a compression level to a turn's content.

        Args:
            turn: the turn to compress.
            level: "verbatim" | "bullet" | "sentence" | "drop".

        Returns:
            The turn with summary/level set ("drop" leaves only a fidelity_ref marker).
        """
        # Step 1: "verbatim" -> leave content unchanged.
        # Step 2: "bullet"   -> Haiku bullet-point summary into turn.summary.
        # Step 3: "sentence" -> Haiku single-sentence summary into turn.summary.
        # Step 4: "drop"     -> no inline text; rely on turn.fidelity_ref for recovery.
        # Step 5: Set turn.level and return it.
        raise NotImplementedError
