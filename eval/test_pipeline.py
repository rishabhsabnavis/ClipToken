"""End-to-end eval for the ContextOS request path and learning loop.

All tests run offline against the fake Anthropic client from ``conftest.py``. Each
test appends a human-readable line to ``results/eval_results.txt`` so the token
reduction and loss numbers can be reviewed after the run.
"""

from __future__ import annotations

import importlib
from datetime import datetime, timezone
from pathlib import Path

from contextos.learning.detector import LossDetector
from contextos.learning.policy import CompressionPolicy
from contextos.learning.store import FidelityStore
from contextos.modules.adaptive import AdaptiveCompressor
from contextos.modules.assembler import ContextAssembler
from contextos.modules.compressor import ToolResultCompressor
from contextos.modules.substitutor import SymbolSubstitutor
from contextos.pipeline import Pipeline
from contextos.schemas.models import MessagesRequest, Session

RESULTS = Path(__file__).parent.parent / "results" / "eval_results.txt"


def _log(line: str) -> None:
    """Append one timestamped result line to the eval results file."""
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with RESULTS.open("a") as fh:
        fh.write(f"[{stamp}] {line}\n")


def _pipeline(fake_client, tmp_path) -> Pipeline:
    """Build a request-path pipeline bound to the fake client (aggressive tiers)."""
    store = FidelityStore(str(tmp_path / "fidelity.db"))
    policy = CompressionPolicy(str(tmp_path / "policy.json"))
    model = "claude-haiku-4-5-20251001"
    return Pipeline(
        compressor=ToolResultCompressor(fake_client, model),
        substitutor=SymbolSubstitutor(),
        adaptive=AdaptiveCompressor(
            fake_client, model, store, coldstart_tier1=1, coldstart_tier2=2
        ),
        assembler=ContextAssembler(),
        policy=policy,
        deduplicator=None,
    )


def test_token_reduction(fake_client, conversation, tmp_path) -> None:
    """The compressed context must be >= 50% smaller than the raw context."""
    pipe = _pipeline(fake_client, tmp_path)
    request = MessagesRequest(**conversation)
    session = Session(session_id="eval-token")

    pipe.process(request, session)

    ratio = 1.0 - session.tokens_after / session.tokens_before
    _log(
        f"token_reduction tokens_before={session.tokens_before} "
        f"tokens_after={session.tokens_after} compression_ratio={ratio:.3f} "
        f"modules={','.join(session.modules_fired)}"
    )
    assert session.tokens_before > 0
    assert ratio >= 0.50, f"expected >=50% reduction, got {ratio:.1%}"


def test_reversibility(fake_client, conversation, tmp_path) -> None:
    """Every compressed turn must be byte-for-byte recoverable from the FidelityStore."""
    store = FidelityStore(str(tmp_path / "fidelity.db"))
    policy = CompressionPolicy(str(tmp_path / "policy.json"))
    model = "claude-haiku-4-5-20251001"
    pipe = Pipeline(
        compressor=ToolResultCompressor(fake_client, model),
        substitutor=SymbolSubstitutor(),
        adaptive=AdaptiveCompressor(
            fake_client, model, store, coldstart_tier1=1, coldstart_tier2=2
        ),
        assembler=ContextAssembler(),
        policy=policy,
        deduplicator=None,
    )
    request = MessagesRequest(**conversation)
    session = Session(session_id="eval-reverse")

    pipe.process(request, session)

    compressed = [t for t in session.turns if t.level != "verbatim"]
    assert compressed, "expected at least one compressed turn to verify recovery"
    for turn in compressed:
        assert turn.fidelity_ref is not None
        recovered = store.get(turn.fidelity_ref)
        assert recovered == turn.content, "FidelityStore did not return the original"
    _log(f"reversibility recovered_turns={len(compressed)} all_byte_exact=True")


def test_detector_rerequest() -> None:
    """A re-run of a tool whose result was compressed produces a loss event."""
    detector = LossDetector()
    session = Session(session_id="eval-detect")
    detector.note_compression(session.session_id, tool_name="web_search", level="drop")

    response = {
        "content": [
            {"type": "text", "text": "I need to search again."},
            {"type": "tool_use", "name": "web_search", "input": {"query": "again"}},
        ]
    }
    events = detector.scan(session, response)

    assert len(events) == 1
    assert events[0].tool_name == "web_search"
    assert events[0].signal == "re_request"
    _log(f"detector_rerequest events={len(events)} signal={events[0].signal}")


def test_learning_loop_closed(fake_client, tmp_path) -> None:
    """Loss on a tool must make the adaptive compressor protect that tool's turns.

    This verifies the end-to-end loop: the AdaptiveCompressor keys its policy decision
    on the same (segment_type, tool_name) the LossDetector blames loss on, so learned
    loss actually changes future compression of that tool.
    """
    store = FidelityStore(str(tmp_path / "fidelity.db"))
    policy = CompressionPolicy(str(tmp_path / "policy.json"))
    model = "claude-haiku-4-5-20251001"
    adaptive = AdaptiveCompressor(
        fake_client, model, store, coldstart_tier1=0, coldstart_tier2=0
    )

    from contextos.schemas.models import LossEvent, Turn

    def tool_turn() -> Turn:
        return Turn(
            index=0,
            role="user",
            content="a sizable tool result that is a candidate for compression " * 8,
            segment_type="tool_result",
            tool_name="web_search",
        )

    # The adaptive compressor must actually derive the tool identity from the turn.
    seg = adaptive._featurize(tool_turn(), [tool_turn()])
    assert seg.tool_name == "web_search" and seg.segment_type == "tool_result"

    # Before learning, the policy compresses this tool's result (lossy).
    before = policy.decide(seg)
    assert before != "verbatim"

    # Feed repeated re-request losses for web_search, as the detector would.
    events = [
        LossEvent("tool_result", "web_search", before, "re_request", 0)
        for _ in range(4)
    ]
    policy.update(events)

    # After learning, the same tool result is protected (kept verbatim on the wire).
    after = policy.decide(adaptive._featurize(tool_turn(), [tool_turn()]))
    assert after == "verbatim", f"policy did not protect a repeatedly-lost tool: {after}"
    _log(f"learning_loop_closed tool=web_search before={before} after={after}")


def test_schema_promotion(tmp_path, monkeypatch, fake_client) -> None:
    """Repeated loss promotes a pruned schema: keep the needle, drop the filler."""
    monkeypatch.setenv("CONTEXTOS_FIDELITY_STORE_PATH", str(tmp_path / "fid.db"))
    monkeypatch.setenv("CONTEXTOS_POLICY_PATH", str(tmp_path / "policy.json"))
    monkeypatch.setenv("CONTEXTOS_SCHEMA_PATH", str(tmp_path / "schemas.json"))
    monkeypatch.setenv("CONTEXTOS_SCHEMA_PROMOTE_THRESHOLD", "2")

    import contextos.main as main
    import contextos.schemas.registry as registry

    registry.schema_registry.clear()  # isolate from other tests
    main = importlib.reload(main)
    state = main._state()

    from contextos.schemas.models import LossEvent

    # A tool result with a short "needle" (a token) and a large "filler" blob.
    token = "a7f3c9e2b1d84f6099aa12ce77bb3401"
    filler = "verbose operational metadata not needed for the task " * 8
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "get_config", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": {"service": "api", "DEPLOY_TOKEN": token, "notes": filler},
                }
            ],
        },
    ]

    assert registry.get_schema("get_config") is None

    # Two re-request losses -> promotion.
    state.policy.update(
        [LossEvent("tool_result", "get_config", "compressor", "re_request", 0)] * 2
    )
    main._promote_schemas(state, messages)

    # Pruned schema keeps the compact needle + short fields, drops the big blob.
    schema = registry.get_schema("get_config")
    assert schema is not None
    assert "DEPLOY_TOKEN" in schema and "service" in schema
    assert "notes" not in schema, "verbose filler field should be pruned"
    assert "get_config" in state.promoted
    assert (tmp_path / "schemas.json").exists()

    # Module 1 now extracts deterministically: exact token kept, filler gone.
    from contextos.modules.compressor import ToolResultCompressor

    out = ToolResultCompressor(fake_client, "claude-haiku-4-5-20251001").compress(
        "get_config", {"service": "api", "DEPLOY_TOKEN": token, "notes": filler}
    )
    assert token in out and "Compact summary" not in out
    assert "verbose operational metadata" not in out

    # Persistence: a fresh registry reloads the promoted schema.
    registry.schema_registry.clear()
    registry.load_registry(str(tmp_path / "schemas.json"))
    assert registry.get_schema("get_config") is not None
    _log(f"schema_promotion tool=get_config fields={schema} persisted=True")


def test_endpoint_roundtrip(conversation, tmp_path, monkeypatch) -> None:
    """The FastAPI surface compresses, forwards, and reports stats end to end."""
    monkeypatch.setenv("CONTEXTOS_FIDELITY_STORE_PATH", str(tmp_path / "fid.db"))
    monkeypatch.setenv("CONTEXTOS_POLICY_PATH", str(tmp_path / "policy.json"))
    monkeypatch.setenv("CONTEXTOS_COLDSTART_TIER1", "1")
    monkeypatch.setenv("CONTEXTOS_COLDSTART_TIER2", "2")

    from fastapi.testclient import TestClient

    import contextos.main as main

    main = importlib.reload(main)  # rebuild AppState with the tmp env

    from eval.conftest import FakeAnthropic

    main.set_client_factory(lambda api_key: FakeAnthropic())
    client = TestClient(main.app)

    headers = {"x-api-key": "sk-test", "contextos-session-id": "eval-endpoint"}
    resp = client.post("/v1/messages", json=conversation, headers=headers)
    assert resp.status_code == 200, resp.text
    assert resp.json()["role"] == "assistant"

    stats = client.get("/v1/stats/eval-endpoint").json()
    assert stats["tokens_before"] > stats["tokens_after"] > 0
    assert stats["compression_ratio"] >= 0.50

    curve = client.get("/v1/loss-curve").json()
    assert len(curve["points"]) >= 1
    _log(
        f"endpoint status=200 compression_ratio={stats['compression_ratio']:.3f} "
        f"loss_events={stats['loss_events']}"
    )
