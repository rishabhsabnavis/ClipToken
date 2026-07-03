"""Data structures shared across ContextOS.

This file is intentionally *structure only*: the dataclasses below define the shape
of data that flows through the pipeline and learning loop. The behavioural logic
lives in the module classes, not here. Fields mirror the contracts in CLAUDE.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# API models (mirror the Anthropic messages API surface)
# ---------------------------------------------------------------------------


class MessagesRequest(BaseModel):
    """Incoming POST /v1/messages body. Mirrors the Anthropic request shape.

    `extra = "allow"` so any Anthropic field we don't model explicitly is still
    forwarded untouched.
    """

    model: str
    max_tokens: int
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    system: str | None = None

    model_config = {"extra": "allow"}


class StatsResponse(BaseModel):
    """Response body for GET /v1/stats/{session_id}."""

    session_id: str
    turns: int
    tokens_before: int
    tokens_after: int
    compression_ratio: float
    modules_fired: list[str]
    loss_events: int
    loss_rate: float
    sessions_seen: int


# ---------------------------------------------------------------------------
# Core dataclasses (the request path operates on these)
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    """One message in the conversation history, plus how it was compressed."""

    index: int
    role: str  # "user" | "assistant"
    content: str
    summary: str | None = None
    level: str = "verbatim"  # "verbatim" | "bullet" | "sentence" | "drop"
    fidelity_ref: str | None = None  # content hash in FidelityStore, if compressed
    # Set by the pipeline so the AdaptiveCompressor can build a Segment whose
    # (segment_type, tool_name) key matches the one the LossDetector attributes loss
    # to -- this is what closes the learning loop end to end.
    segment_type: str | None = None  # "tool_result" | "assistant_reasoning" | "user"
    tool_name: str | None = None  # tool that produced this turn's content, if any


@dataclass
class Segment:
    """Features the CompressionPolicy uses to decide a compression level for a Turn."""

    turn_index: int
    age_turns: int
    token_len: int
    tool_name: str | None
    semantic_relevance: float  # cosine sim to current goal / last user turn
    times_referenced_recently: int
    segment_type: str  # "tool_result" | "assistant_reasoning" | "user" | ...


@dataclass
class LossEvent:
    """A detected instance where a past compression discarded something the agent needed."""

    segment_type: str
    tool_name: str | None
    level: str  # the compression level that caused the loss
    signal: str  # "re_request" | "shadow_divergence" | "judge"
    turn_index: int


@dataclass
class Session:
    """Per-session state, keyed by the contextos-session-id header.

    Holds turn history (used by the AdaptiveCompressor) and running counters
    (surfaced by GET /v1/stats).
    """

    session_id: str
    turns: list[Turn] = field(default_factory=list)
    tokens_before: int = 0
    tokens_after: int = 0
    loss_events: int = 0
    sessions_seen: int = 0
    modules_fired: list[str] = field(default_factory=list)
