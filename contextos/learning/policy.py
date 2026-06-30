"""Layer C -- CompressionPolicy (learn, get less lossy over time).

The policy maps a :class:`Segment` (features describing one turn) to a compression
*level*. It picks the most aggressive level whose *predicted task-impact* stays under
``CONTEXTOS_IMPACT_BUDGET``. As the LossDetector feeds it loss events, the policy
raises the protection of the ``(segment_type, tool_name)`` combinations that have
caused loss -- so the same mistake is made less often, and loss falls over time.

MVP form (this file): online-updated per-``(segment_type, tool_name)`` loss-propensity
counts feeding a thresholded decision. Upgrade path: a small sklearn
logistic-regression / contextual bandit once enough data exists.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contextos.schemas.models import LossEvent, Segment

logger = logging.getLogger(__name__)

# Compression levels ordered from most aggressive (lossiest) to least.
_LEVELS_AGGRESSIVE_FIRST: tuple[str, ...] = ("drop", "sentence", "bullet", "verbatim")

# Rough base task-impact of each level before per-segment adjustment. "verbatim"
# is lossless by definition; the rest lose progressively more detail.
_BASE_IMPACT: dict[str, float] = {
    "verbatim": 0.0,
    "bullet": 0.08,
    "sentence": 0.20,
    "drop": 0.50,
}


class CompressionPolicy:
    """Decide a per-segment compression level and learn from observed loss events."""

    def __init__(self, path: str, impact_budget: float = 0.15) -> None:
        """Create a policy, loading prior learning from disk if present.

        Args:
            path: where the learned state is persisted (``CONTEXTOS_POLICY_PATH``).
            impact_budget: max predicted task-impact the policy will accept
                (``CONTEXTOS_IMPACT_BUDGET``).
        """
        self.path = path
        self.impact_budget = impact_budget
        # Learned state: (segment_type, tool_name) -> number of loss events seen.
        # More loss events => higher predicted impact => less aggressive compression.
        self.loss_counts: dict[tuple[str, str | None], int] = {}
        self.load()

    @property
    def has_data(self) -> bool:
        """Whether the policy has learned anything yet (else callers cold-start)."""
        return bool(self.loss_counts)

    def decide(self, segment: Segment) -> str:
        """Return the most aggressive level whose predicted impact stays under budget.

        Args:
            segment: features describing the turn being compressed.

        Returns:
            One of "drop" | "sentence" | "bullet" | "verbatim". "verbatim" always
            qualifies (impact 0), so a level is always returned.
        """
        for level in _LEVELS_AGGRESSIVE_FIRST:
            if self.predicted_impact(segment, level) <= self.impact_budget:
                return level
        return "verbatim"

    def predicted_impact(self, segment: Segment, level: str) -> float:
        """Estimate the task-impact of applying `level` to `segment`.

        The estimate scales the level's base impact by how relevant the content
        still is and by how often this kind of segment has caused loss before.

        Args:
            segment: features of the turn.
            level: the candidate compression level.

        Returns:
            A non-negative predicted-impact score (comparable to impact_budget).
        """
        base = _BASE_IMPACT.get(level, 0.0)
        # More relevant content hurts more when compressed (range ~0.5..1.5).
        relevance_factor = 0.5 + segment.semantic_relevance
        # Each past loss event for this (type, tool) inflates predicted impact,
        # nudging the policy toward gentler levels for that combination.
        key = (segment.segment_type, segment.tool_name)
        propensity_factor = 1.0 + 0.5 * self.loss_counts.get(key, 0)
        return base * relevance_factor * propensity_factor

    def update(self, events: list[LossEvent]) -> None:
        """Learn from detected loss events by protecting the segments that caused them.

        Args:
            events: loss events emitted by the LossDetector for the last call.
        """
        for event in events:
            key = (event.segment_type, event.tool_name)
            self.loss_counts[key] = self.loss_counts.get(key, 0) + 1
            logger.info(
                "policy.update segment_type=%s tool_name=%s level=%s count=%d",
                event.segment_type,
                event.tool_name,
                event.level,
                self.loss_counts[key],
            )
        if events:
            self.save()

    def save(self) -> None:
        """Persist learned state to ``self.path`` (tuple keys flattened to strings)."""
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        serialisable = {
            f"{seg_type}|{tool or ''}": count
            for (seg_type, tool), count in self.loss_counts.items()
        }
        Path(self.path).write_text(json.dumps(serialisable, indent=2))

    def load(self) -> None:
        """Load learned state from ``self.path`` if it exists (else start empty)."""
        path = Path(self.path)
        if not path.exists():
            return
        raw: dict[str, int] = json.loads(path.read_text())
        self.loss_counts = {}
        for flat_key, count in raw.items():
            seg_type, tool = flat_key.split("|", 1)
            self.loss_counts[(seg_type, tool or None)] = count
