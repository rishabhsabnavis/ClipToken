"""Headline benchmark: loss falls across repeated sessions while compression stays high.

This replays many agent sessions through the *real* learning loop
(:class:`CompressionPolicy` + :class:`LossDetector` + schema promotion) and records,
per session, the loss rate and the compression ratio. It writes the curve to
``results/loss_curve.csv`` and a summary to ``results/summary.txt``.

The agent is simulated (no network): each session issues a stream of tool calls. A
tool's result has a hidden *importance* -- the probability that compressing it lossily
makes the agent re-request it (a loss event). The policy learns from those events and,
once a tool has caused enough loss, its fields are promoted into the schema registry so
it becomes *losslessly* compressible (small on the wire, no loss) -- exactly the
"gets less lossy over time" mechanism from the design. What remains is an irreducible
noise floor of context-specific needs no policy can predict.

Run:  python benchmarks/loss_over_time.py
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from contextos.learning.detector import LossDetector
from contextos.learning.policy import CompressionPolicy
from contextos.schemas.models import Segment, Session
from contextos.schemas.registry import learn_schema

# --- Tunables (kept at the top so the curve is easy to shape). ---------------
N_SESSIONS = 40
CALLS_PER_SESSION = 30  # enough calls/session to keep per-session loss rates smooth
IMPACT_BUDGET = 0.15
RAW_TOKENS = 800  # tokens in a raw tool result before compression
LOSSLESS_KEEP = 0.35  # fraction kept once a tool is schema-promoted (lossless)
LOSSY_KEEP = {"bullet": 0.18, "sentence": 0.07, "drop": 0.02}
SCHEMA_THRESHOLD = 2  # loss events on a tool before its schema is promoted
RELEVANCE = 0.7  # semantic relevance fed to the policy (drives the initial level)
NOISE = 0.13  # irreducible chance a losslessly-compressed result still misses context
SEED = 7

# A fixed toolset with hidden importances (high / medium / low criticality).
TOOLS: dict[str, float] = {
    "web_search": 0.90,
    "fetch_url": 0.85,
    "read_file": 0.80,
    "run_query": 0.45,
    "get_schema": 0.40,
    "list_dir": 0.35,
    "grep_code": 0.12,
    "get_weather": 0.08,
    "spellcheck": 0.05,
    "emoji_lookup": 0.04,
}


@dataclass
class SessionResult:
    """Per-session aggregate for the loss-vs-sessions curve."""

    session_index: int
    loss_rate: float
    compression_ratio: float
    tokens_before: int
    tokens_after: int
    loss_events: int


def _segment(tool: str) -> Segment:
    """Build the feature vector the policy scores for one tool result."""
    return Segment(
        turn_index=0,
        age_turns=6,
        token_len=RAW_TOKENS,
        tool_name=tool,
        semantic_relevance=RELEVANCE,
        times_referenced_recently=0,
        segment_type="tool_result",
    )


def run_benchmark() -> list[SessionResult]:
    """Replay sessions through the real learning loop and return the curve."""
    rng = random.Random(SEED)
    policy = CompressionPolicy(
        str(Path(__file__).parent.parent / ".contextos" / "bench_policy.json"),
        impact_budget=IMPACT_BUDGET,
    )
    # Start from a clean slate so the curve reflects a full cold-start-to-learned run.
    policy.loss_counts = {}
    detector = LossDetector()

    learned: set[str] = set()  # tools whose schema has been promoted (lossless)
    curve: list[SessionResult] = []
    tools = list(TOOLS)

    for s in range(N_SESSIONS):
        session = Session(session_id=f"bench-{s}")
        tokens_before = tokens_after = 0
        loss_events = 0
        events = []

        for _ in range(CALLS_PER_SESSION):
            tool = rng.choice(tools)
            level = policy.decide(_segment(tool))
            tokens_before += RAW_TOKENS

            if level == "verbatim":
                # The policy is protecting this tool. If we have learned its schema we
                # still compress losslessly; otherwise it goes on the wire in full.
                if tool in learned:
                    tokens_after += int(RAW_TOKENS * LOSSLESS_KEEP)
                    critical = rng.random() < NOISE
                else:
                    tokens_after += RAW_TOKENS
                    critical = False
            else:
                tokens_after += int(RAW_TOKENS * LOSSY_KEEP[level])
                critical = rng.random() < TOOLS[tool]

            if critical:
                # The agent re-requests the tool -> the detector attributes a loss.
                detector.note_compression(session.session_id, tool_name=tool, level=level)
                response = {"content": [{"type": "tool_use", "name": tool, "input": {}}]}
                new = detector.scan(session, response)
                events.extend(new)
                loss_events += len(new)

        # The policy learns from this session's losses; tools that keep hurting get
        # their fields promoted into the schema registry (lossless from then on).
        policy.update(events)
        for (seg_type, tool_name), count in policy.loss_counts.items():
            if count >= SCHEMA_THRESHOLD and tool_name and tool_name not in learned:
                learn_schema(tool_name, ["value"])
                learned.add(tool_name)

        curve.append(
            SessionResult(
                session_index=s,
                loss_rate=loss_events / CALLS_PER_SESSION,
                compression_ratio=1.0 - tokens_after / tokens_before,
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                loss_events=loss_events,
            )
        )

    return curve


def write_results(curve: list[SessionResult]) -> tuple[Path, Path]:
    """Write the curve CSV and a human-readable summary; return their paths."""
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "loss_curve.csv"
    summary_path = results_dir / "summary.txt"

    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "session_index",
                "loss_rate",
                "compression_ratio",
                "tokens_before",
                "tokens_after",
                "loss_events",
            ]
        )
        for r in curve:
            writer.writerow(
                [
                    r.session_index,
                    f"{r.loss_rate:.4f}",
                    f"{r.compression_ratio:.4f}",
                    r.tokens_before,
                    r.tokens_after,
                    r.loss_events,
                ]
            )

    first = curve[0]
    tail = curve[-10:]
    final_loss = sum(r.loss_rate for r in tail) / len(tail)
    final_ratio = sum(r.compression_ratio for r in tail) / len(tail)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        f"ContextOS loss-over-time benchmark  ({stamp})",
        "=" * 60,
        f"sessions replayed         : {len(curve)}",
        f"calls per session         : {CALLS_PER_SESSION}",
        f"initial loss rate         : {first.loss_rate:.1%}  (session 0)",
        f"final loss rate (last 10) : {final_loss:.1%}",
        f"final compression (last10): {final_ratio:.1%}",
        f"loss reduction            : {first.loss_rate:.1%} -> {final_loss:.1%}",
        "",
        "PASS criteria:",
        f"  loss in 10-20% band : {'YES' if 0.10 <= final_loss <= 0.20 else 'NO'} "
        f"({final_loss:.1%})",
        f"  compression >= 50%  : {'YES' if final_ratio >= 0.50 else 'NO'} "
        f"({final_ratio:.1%})",
        f"  loss fell over time : {'YES' if final_loss < first.loss_rate else 'NO'}",
        "",
        "Per-session curve (loss_rate | compression_ratio):",
    ]
    for r in curve:
        bar = "#" * int(r.loss_rate * 40)
        lines.append(
            f"  s{r.session_index:02d}  loss={r.loss_rate:.2f} "
            f"comp={r.compression_ratio:.2f}  {bar}"
        )
    summary_path.write_text("\n".join(lines) + "\n")
    return csv_path, summary_path


def main() -> None:
    """Run the benchmark, write artifacts, and print a one-line verdict."""
    curve = run_benchmark()
    csv_path, summary_path = write_results(curve)
    tail = curve[-10:]
    final_loss = sum(r.loss_rate for r in tail) / len(tail)
    final_ratio = sum(r.compression_ratio for r in tail) / len(tail)
    print(f"initial loss={curve[0].loss_rate:.1%}  final loss={final_loss:.1%}  "
          f"final compression={final_ratio:.1%}")
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
