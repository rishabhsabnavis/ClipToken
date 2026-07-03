"""FastAPI entry point for ContextOS.

Exposes an Anthropic-compatible ``POST /v1/messages`` surface plus stats endpoints.
An agent points its base URL here instead of at Anthropic; ContextOS compresses the
request context (Modules 1-5), forwards the compressed call to the real provider,
watches the response for loss (LossDetector), and lets the CompressionPolicy learn --
all transparently, returning the provider response unchanged.

Wiring notes:
* Heavy dependencies (the embedder) are optional and off by default so the server can
  start without downloading models; set ``CONTEXTOS_ENABLE_DEDUP=1`` to enable Module 2.
* The Anthropic client is built per request from the forwarded ``x-api-key`` header.
  Tests substitute a fake via :func:`set_client_factory`, so the whole app runs offline.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from fastapi import FastAPI, Header, HTTPException

from contextos.learning.detector import LossDetector
from contextos.learning.policy import CompressionPolicy
from contextos.learning.store import FidelityStore
from contextos.modules.adaptive import AdaptiveCompressor
from contextos.modules.assembler import ContextAssembler
from contextos.modules.compressor import ToolResultCompressor
from contextos.modules.deduplicator import SemanticDeduplicator
from contextos.modules.substitutor import SymbolSubstitutor
from contextos.pipeline import Pipeline
from contextos.schemas.models import MessagesRequest, Session, StatsResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings (env vars from CLAUDE.md, with defaults)
# ---------------------------------------------------------------------------


@dataclass
class Settings:
    """Runtime configuration, read from environment variables."""

    port: int = int(os.getenv("CONTEXTOS_PORT", "8000"))
    dedup_threshold: float = float(os.getenv("CONTEXTOS_DEDUP_THRESHOLD", "0.92"))
    symbol_min_length: int = int(os.getenv("CONTEXTOS_SYMBOL_MIN_LENGTH", "20"))
    symbol_min_occurrences: int = int(os.getenv("CONTEXTOS_SYMBOL_MIN_OCCURRENCES", "2"))
    impact_budget: float = float(os.getenv("CONTEXTOS_IMPACT_BUDGET", "0.15"))
    shadow_sample_rate: float = float(os.getenv("CONTEXTOS_SHADOW_SAMPLE_RATE", "0.05"))
    fidelity_store_path: str = os.getenv(
        "CONTEXTOS_FIDELITY_STORE_PATH", "./.contextos/fidelity.db"
    )
    policy_path: str = os.getenv("CONTEXTOS_POLICY_PATH", "./.contextos/policy.json")
    schema_path: str = os.getenv("CONTEXTOS_SCHEMA_PATH", "./.contextos/schemas.json")
    schema_promote_threshold: int = int(
        os.getenv("CONTEXTOS_SCHEMA_PROMOTE_THRESHOLD", "2")
    )
    coldstart_tier1: int = int(os.getenv("CONTEXTOS_COLDSTART_TIER1", "5"))
    coldstart_tier2: int = int(os.getenv("CONTEXTOS_COLDSTART_TIER2", "10"))
    compress_model: str = os.getenv(
        "CONTEXTOS_COMPRESS_MODEL", "claude-haiku-4-5-20251001"
    )
    enable_dedup: bool = os.getenv("CONTEXTOS_ENABLE_DEDUP", "0") == "1"


# ---------------------------------------------------------------------------
# Application state (created once, shared across requests)
# ---------------------------------------------------------------------------


@dataclass
class LossPoint:
    """One entry in the loss-vs-sessions curve (the headline artifact)."""

    session_index: int
    loss_rate: float
    compression_ratio: float


@dataclass
class AppState:
    """Long-lived, cross-request state: sessions, learning loop, and the curve."""

    settings: Settings
    store: FidelityStore
    policy: CompressionPolicy
    detector: LossDetector
    sessions: dict[str, Session] = field(default_factory=dict)
    loss_curve: list[LossPoint] = field(default_factory=list)
    embedder: Any | None = None
    # Tools whose schema has been auto-promoted into the registry (so Module 1
    # compresses them deterministically/losslessly instead of via a Haiku summary).
    promoted: set[str] = field(default_factory=set)


# The client factory is overridable so tests can inject a fake Anthropic client.
ClientFactory = Callable[[str], Any]


def _default_client_factory(api_key: str) -> Any:
    """Build a real Anthropic client from a forwarded API key."""
    from anthropic import Anthropic

    return Anthropic(api_key=api_key)


_client_factory: ClientFactory = _default_client_factory


def set_client_factory(factory: ClientFactory) -> None:
    """Override how Anthropic clients are built (used by tests to inject a fake)."""
    global _client_factory
    _client_factory = factory


def _build_state() -> AppState:
    """Construct the shared application state from settings."""
    settings = Settings()
    # Resume any schemas the learning loop promoted in previous runs.
    from contextos.schemas.registry import load_registry

    load_registry(settings.schema_path)
    store = FidelityStore(settings.fidelity_store_path)
    policy = CompressionPolicy(settings.policy_path, impact_budget=settings.impact_budget)
    detector = LossDetector(shadow_sample_rate=settings.shadow_sample_rate)
    state = AppState(settings=settings, store=store, policy=policy, detector=detector)
    if settings.enable_dedup:
        from sentence_transformers import SentenceTransformer

        state.embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return state


app = FastAPI(title="ContextOS", version="0.1.0")
app.state.ctx = _build_state()


def _state() -> AppState:
    """Return the shared AppState attached to the app."""
    return app.state.ctx


def _build_pipeline(client: Any, state: AppState) -> Pipeline:
    """Assemble a Pipeline bound to a specific Anthropic client."""
    s = state.settings
    deduplicator = (
        SemanticDeduplicator(state.embedder) if state.embedder is not None else None
    )
    return Pipeline(
        compressor=ToolResultCompressor(client, s.compress_model),
        substitutor=SymbolSubstitutor(),
        adaptive=AdaptiveCompressor(
            client,
            s.compress_model,
            state.store,
            embedder=state.embedder,
            coldstart_tier1=s.coldstart_tier1,
            coldstart_tier2=s.coldstart_tier2,
        ),
        assembler=ContextAssembler(),
        policy=state.policy,
        deduplicator=deduplicator,
        dedup_threshold=s.dedup_threshold,
        symbol_min_length=s.symbol_min_length,
        symbol_min_occurrences=s.symbol_min_occurrences,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/v1/messages")
def messages(
    request: MessagesRequest,
    x_api_key: str | None = Header(default=None),
    contextos_session_id: str = Header(default="default"),
) -> dict:
    """Compress the request, forward it to Anthropic, and learn from the response.

    Args:
        request: the Anthropic-style messages request.
        x_api_key: the caller's Anthropic key, forwarded to the provider.
        contextos_session_id: keys per-session state (turn history + learning loop).

    Returns:
        The provider response, unchanged.
    """
    state = _state()
    # Prefer the forwarded header; fall back to the server's own env var so local
    # testing works with a bare `export ANTHROPIC_API_KEY=...` and no custom headers.
    api_key = x_api_key or os.getenv("ANTHROPIC_API_KEY")
    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="missing API key: send x-api-key header or set ANTHROPIC_API_KEY",
        )

    # Per-session state (persists turn history + counters across calls).
    session = state.sessions.get(contextos_session_id)
    if session is None:
        session = Session(session_id=contextos_session_id)
        state.sessions[contextos_session_id] = session
    session.sessions_seen += 1

    client = _client_factory(api_key)
    pipeline = _build_pipeline(client, state)

    # Tell the detector which tool *calls* had their results compressed this call, so
    # a later re-run of the same call can be attributed to that compression.
    for tool_name, args in _compressed_tool_calls(request.messages):
        state.detector.note_compression(
            session.session_id, tool_name=tool_name, level="compressor", args=args
        )

    # Request path: compress the context.
    compressed = pipeline.process(request, session)

    # Lift any assembler-emitted system legend into the top-level `system` field,
    # since the Anthropic API takes system separately from messages.
    system, wire_messages = _split_system(compressed, request.system)

    # Forward the compressed call to the provider.
    create_kwargs: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "messages": wire_messages,
    }
    if system:
        create_kwargs["system"] = system
    if request.tools:
        create_kwargs["tools"] = request.tools
    raw = client.messages.create(**create_kwargs)
    response = _to_dict(raw)

    # Learning loop: detect loss and let the policy learn from it.
    events = state.detector.scan(session, response)
    if events:
        session.loss_events += len(events)
        state.policy.update(events)
        # Tools that keep causing loss get their fields promoted into the schema
        # registry, so Module 1 compresses them deterministically (no lossy Haiku
        # summary) from now on -- the "gets less lossy over time" mechanism.
        _promote_schemas(state, request.messages)

    _record_curve(state, session)
    return response


@app.get("/v1/stats/{session_id}")
def stats(session_id: str) -> StatsResponse:
    """Return compression + learning stats for one session."""
    state = _state()
    session = state.sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"unknown session {session_id}")

    ratio = _ratio(session.tokens_before, session.tokens_after)
    turns = len(session.turns)
    loss_rate = (session.loss_events / turns) if turns else 0.0
    return StatsResponse(
        session_id=session_id,
        turns=turns,
        tokens_before=session.tokens_before,
        tokens_after=session.tokens_after,
        compression_ratio=ratio,
        modules_fired=session.modules_fired,
        loss_events=session.loss_events,
        loss_rate=loss_rate,
        sessions_seen=session.sessions_seen,
    )


@app.get("/v1/loss-curve")
def loss_curve() -> dict:
    """Return the headline artifact: loss_rate + compression_ratio per session index."""
    state = _state()
    return {
        "points": [
            {
                "session_index": p.session_index,
                "loss_rate": p.loss_rate,
                "compression_ratio": p.compression_ratio,
            }
            for p in state.loss_curve
        ]
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool_use_index(messages: list[dict]) -> dict[str, dict[str, Any]]:
    """Map each tool_use id to its {name, input} from the assistant tool_use blocks."""
    index: dict[str, dict[str, Any]] = {}
    for message in messages:
        for block in message.get("content") or []:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and block.get("id")
            ):
                index[block["id"]] = {
                    "name": block.get("name", "unknown"),
                    "input": block.get("input", {}),
                }
    return index


def _compressed_tool_calls(messages: list[dict]) -> list[tuple[str, Any]]:
    """Return (tool_name, args) for every tool result present in the messages.

    Every tool result is compressed by Module 1, so each of these is a call whose
    result may later be re-requested if the compression dropped what the agent needed.
    """
    index = _tool_use_index(messages)
    calls: list[tuple[str, Any]] = []
    for message in messages:
        for block in message.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                meta = index.get(block.get("tool_use_id"), {"name": "unknown", "input": {}})
                calls.append((meta["name"], meta["input"]))
    return calls


def _promote_schemas(state: AppState, messages: list[dict]) -> None:
    """Promote schemas for tools that have crossed the loss threshold.

    A tool that keeps causing re-request losses is one we must stop paraphrasing.
    We register the fields observed in its results so Module 1 switches to
    deterministic field extraction (exact, recoverable) instead of a lossy Haiku
    summary, and persist the registry so this survives restarts.
    """
    from contextos.schemas.registry import learn_schema, save_registry

    threshold = state.settings.schema_promote_threshold
    changed = False
    for (segment_type, tool_name), count in state.policy.loss_counts.items():
        if segment_type != "tool_result" or not tool_name:
            continue
        if count < threshold or tool_name in state.promoted:
            continue
        fields = _observed_fields(tool_name, messages)
        if not fields:
            continue
        learn_schema(tool_name, fields)
        state.promoted.add(tool_name)
        changed = True
        logger.info("promoted schema tool=%s fields=%s", tool_name, fields)
    if changed:
        save_registry(state.settings.schema_path)


def _observed_fields(
    tool_name: str, messages: list[dict], small_chars: int = 160
) -> list[str]:
    """Return the fields worth keeping for a tool -- the "safe" schema to promote.

    A field is kept if it is *compact* (a short scalar / id / token the agent may
    need exactly) or its value is *referenced downstream* (so it clearly mattered).
    Large, unreferenced blobs (verbose ``notes``, huge nested payloads) are dropped,
    so promotion protects the needle while still compressing the noise.

    Missing-per-result fields are handled safely by the compressor's ``.get``.
    """
    index = _tool_use_index(messages)
    downstream = _all_text(messages)

    fields: list[str] = []
    for message in messages:
        for block in message.get("content") or []:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            if index.get(block.get("tool_use_id"), {}).get("name") != tool_name:
                continue
            result = block.get("content")
            if not isinstance(result, dict):
                continue
            for key, value in result.items():
                serialized = value if isinstance(value, str) else json.dumps(value, default=str)
                keep = len(serialized) <= small_chars or _value_referenced(serialized, downstream)
                if keep and key not in fields:
                    fields.append(key)
    return fields


def _all_text(messages: list[dict]) -> str:
    """Concatenate all plain text across the messages (for reference checks)."""
    parts: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
    return "\n".join(parts)


def _value_referenced(serialized: str, downstream: str, probe: int = 16) -> bool:
    """Whether a field's value appears to be cited downstream (handles long values)."""
    sample = serialized.strip().strip('"')
    if not sample:
        return False
    if len(sample) <= probe:
        return sample in downstream
    return sample[:probe] in downstream


def _split_system(
    messages: list[dict], base_system: str | None
) -> tuple[str | None, list[dict]]:
    """Separate any system-role messages from the wire messages.

    The assembler prepends the symbol legend as a system-role message; the Anthropic
    API expects `system` as a top-level string, so fold those out here.
    """
    system_parts: list[str] = []
    if base_system:
        system_parts.append(base_system)
    wire: list[dict] = []
    for message in messages:
        if message.get("role") == "system":
            system_parts.append(str(message.get("content", "")))
        else:
            wire.append(message)
    system = "\n\n".join(p for p in system_parts if p) or None
    return system, wire


def _to_dict(raw: Any) -> dict:
    """Coerce a provider response (pydantic model or dict) into a plain dict."""
    if isinstance(raw, dict):
        return raw
    if hasattr(raw, "model_dump"):
        return raw.model_dump()
    if hasattr(raw, "dict"):
        return raw.dict()
    return dict(raw)


def _ratio(before: int, after: int) -> float:
    """Fraction of tokens saved (0.0 when there were none to begin with)."""
    return 1.0 - (after / before) if before else 0.0


def _record_curve(state: AppState, session: Session) -> None:
    """Append a loss-curve point for the current session state."""
    turns = len(session.turns)
    loss_rate = (session.loss_events / turns) if turns else 0.0
    state.loss_curve.append(
        LossPoint(
            session_index=len(state.loss_curve),
            loss_rate=loss_rate,
            compression_ratio=_ratio(session.tokens_before, session.tokens_after),
        )
    )


def run() -> None:
    """Console-script entry point (``contextos``): serve the app with uvicorn."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=_state().settings.port)


if __name__ == "__main__":
    run()
