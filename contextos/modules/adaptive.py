"""Module 4 -- AdaptiveCompressor (the differentiator).

Replaces the old fixed-tier "summarization pyramid." Instead of compressing turns by
age, it compresses each turn by its *predicted impact on the agent's next action*,
choosing the most aggressive level whose predicted impact stays under a budget.

Before compressing anything it writes the original to the FidelityStore, so every
decision is reversible -- this is what lets loss fall toward zero over time.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tiktoken import get_encoding

if TYPE_CHECKING:
    from anthropic import Anthropic
    from sentence_transformers import SentenceTransformer

    from contextos.learning.policy import CompressionPolicy
    from contextos.learning.store import FidelityStore
    from contextos.schemas.models import Segment, Turn

logger = logging.getLogger(__name__)

# Number of most-recent turns scanned when counting how often a turn is referenced.
_RECENT_WINDOW = 3
# Prefix length used as a cheap "fingerprint" when detecting references.
_FINGERPRINT_CHARS = 40


class AdaptiveCompressor:
    """Decide and apply a per-turn compression level using the learned policy."""

    def __init__(
        self,
        client: Anthropic,
        model: str,
        store: FidelityStore,
        embedder: SentenceTransformer | None = None,
        coldstart_tier1: int = 5,
        coldstart_tier2: int = 10,
    ) -> None:
        """Wire up dependencies and the cold-start age prior.

        Args:
            client: Anthropic client for Haiku summaries.
            model: Haiku model id.
            store: FidelityStore, written to before any lossy compression.
            embedder: optional SentenceTransformer for semantic_relevance; if None,
                relevance falls back to a neutral 0.5 (no semantic signal).
            coldstart_tier1: age boundary below which turns stay verbatim.
            coldstart_tier2: age boundary below which turns get a bullet summary.
        """
        # Step 1: Save client, model, store, embedder, and the two cold-start boundaries.
        self.client = client
        self.model = model
        self.store = store
        self.embedder = embedder
        self.coldstart_tier1 = coldstart_tier1
        self.coldstart_tier2 = coldstart_tier2
        # tiktoken only knows OpenAI encodings; cl100k_base is a good-enough proxy
        # for the before/after token logging (same approach as the compressor).
        self.encoding = get_encoding("cl100k_base")

    def compress_history(
        self, turns: list[Turn], policy: CompressionPolicy
    ) -> list[Turn]:
        """Compress each turn to the level the policy chooses (or the cold-start prior).

        Args:
            turns: full session turn history (oldest first).
            policy: the CompressionPolicy that maps a Segment -> level.

        Returns:
            The same turns with `level`, `summary`, and `fidelity_ref` populated.
        """
        for turn in turns:
            # Step 1: Describe the turn as a Segment of features.
            segment = self._featurize(turn, turns)

            # Step 2: Ask the policy for a level; cold-start until it has data.
            if policy.has_data:
                level = policy.decide(segment)
                predicted: float | None = policy.predicted_impact(segment, level)
            else:
                level = self._coldstart_level(segment)
                predicted = None

            tokens_before = len(self.encoding.encode(turn.content))

            # Step 3: Anything lossy is written to the FidelityStore first, so the
            #         original is recoverable from the hash on turn.fidelity_ref.
            if level != "verbatim":
                turn.fidelity_ref = self.store.put(turn.content)

            # Step 4: Apply the chosen level (may call Haiku for summaries).
            self._apply_level(turn, level)

            # Step 5: Log the decision and token deltas.
            if level == "verbatim":
                tokens_after = tokens_before
            else:
                tokens_after = len(self.encoding.encode(turn.summary or ""))
            logger.info(
                "adaptive turn=%d level=%s predicted_impact=%s tokens_before=%d tokens_after=%d",
                turn.index,
                level,
                f"{predicted:.3f}" if predicted is not None else "coldstart",
                tokens_before,
                tokens_after,
            )

        # Step 6: Return the updated turns.
        return turns

    def _featurize(self, turn: Turn, turns: list[Turn]) -> Segment:
        """Build the feature vector the policy needs for one turn.

        Args:
            turn: the turn being scored.
            turns: full history (for age, references, relevance to the latest goal).

        Returns:
            A Segment describing this turn.
        """
        from contextos.schemas.models import Segment

        # Step 1: age = distance from the newest turn (newest has age 0).
        age_turns = len(turns) - 1 - turn.index
        # Step 2: approximate token length of the content.
        token_len = len(self.encoding.encode(turn.content))
        # Step 3: infer a coarse segment_type / tool_name from the turn shape.
        segment_type, tool_name = self._classify(turn)
        # Step 4: how relevant the content still is to the latest user goal.
        semantic_relevance = self._relevance(turn, turns)
        # Step 5: how often later turns appear to refer back to this content.
        times_referenced_recently = self._reference_count(turn, turns)
        # Step 6: assemble the Segment.
        return Segment(
            turn_index=turn.index,
            age_turns=age_turns,
            token_len=token_len,
            tool_name=tool_name,
            semantic_relevance=semantic_relevance,
            times_referenced_recently=times_referenced_recently,
            segment_type=segment_type,
        )

    def _coldstart_level(self, segment: Segment) -> str:
        """Age-based fallback level, used before the policy has learned anything.

        Recent turns are kept intact and older turns are compressed progressively
        harder -- the standard "summarize old context, keep recent verbatim" prior.

        Args:
            segment: the segment being scored.

        Returns:
            "verbatim" | "bullet" | "sentence" per the cold-start tiers.
        """
        # Step 1: recent (age <= tier1) -> keep verbatim.
        if segment.age_turns <= self.coldstart_tier1:
            return "verbatim"
        # Step 2: middle-aged (age <= tier2) -> light bullet summary.
        if segment.age_turns <= self.coldstart_tier2:
            return "bullet"
        # Step 3: oldest -> aggressive single-sentence summary.
        return "sentence"

    def _apply_level(self, turn: Turn, level: str) -> Turn:
        """Apply a compression level to a turn's content.

        Args:
            turn: the turn to compress.
            level: "verbatim" | "bullet" | "sentence" | "drop".

        Returns:
            The turn with summary/level set ("drop" leaves only a fidelity_ref marker).
        """
        # Step 1: verbatim -> leave content unchanged.
        if level == "verbatim":
            turn.summary = None
        # Step 2: bullet -> Haiku bullet-point summary.
        elif level == "bullet":
            turn.summary = self._summarize(turn.content, style="bullet")
        # Step 3: sentence -> Haiku single-sentence summary.
        elif level == "sentence":
            turn.summary = self._summarize(turn.content, style="sentence")
        # Step 4: drop -> no inline text; rely on turn.fidelity_ref for recovery.
        elif level == "drop":
            turn.summary = None
        else:
            raise ValueError(f"unknown compression level: {level!r}")

        # Step 5: record the level and return the turn.
        turn.level = level
        return turn

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify(self, turn: Turn) -> tuple[str, str | None]:
        """Infer a coarse (segment_type, tool_name) from a turn.

        Turn content here is a plain string, so this is a heuristic: assistant
        turns are reasoning, user turns are user input. Structured tool-result
        detection is left to the pipeline, which has the raw message blocks.
        """
        if turn.role == "assistant":
            return "assistant_reasoning", None
        return "user", None

    def _relevance(self, turn: Turn, turns: list[Turn]) -> float:
        """Cosine similarity of this turn to the latest user goal (0.5 if no embedder)."""
        if self.embedder is None:
            return 0.5
        goal = self._latest_user_content(turns)
        if goal is None or not turn.content.strip():
            return 0.5
        from sklearn.metrics.pairwise import cosine_similarity

        vectors = self.embedder.encode([turn.content, goal])
        sim = float(cosine_similarity([vectors[0]], [vectors[1]])[0][0])
        # Clamp into [0, 1] -- cosine can dip slightly negative for unrelated text.
        return max(0.0, min(1.0, sim))

    def _reference_count(self, turn: Turn, turns: list[Turn]) -> int:
        """Count how many of the most recent turns appear to cite this turn's content."""
        fingerprint = turn.content.strip()[:_FINGERPRINT_CHARS]
        if not fingerprint:
            return 0
        recent = turns[-_RECENT_WINDOW:]
        return sum(
            1
            for other in recent
            if other.index != turn.index and fingerprint in other.content
        )

    def _latest_user_content(self, turns: list[Turn]) -> str | None:
        """Return the content of the most recent user turn, or None if there is none."""
        for turn in reversed(turns):
            if turn.role == "user" and turn.content.strip():
                return turn.content
        return None

    def _summarize(self, content: str, style: str) -> str:
        """Ask Haiku for a bullet or single-sentence summary of one turn.

        Args:
            content: the turn content to compress.
            style: "bullet" or "sentence".

        Returns:
            The Haiku-generated summary string.
        """
        if style == "bullet":
            instruction = "Summarize as a few terse bullet points"
            max_tokens = 120
        else:
            instruction = "Summarize in a single short sentence"
            max_tokens = 60
        # Keep the prompt compact per the project's <200 input-token convention.
        prompt = (
            f"{instruction}, keeping only what an agent needs to continue its "
            f"task:\n{content}"
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")
