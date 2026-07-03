"""Live agent loop: real model, real tools, routed through the ContextOS HTTP server.

This is the real-traffic version of ``loss_over_time.py``. Instead of a simulated
agent, it starts the actual ContextOS FastAPI server in-process, points a real
Anthropic client at it (drop-in: just a ``base_url`` change), and runs a genuine
agentic loop across several sessions.

The task is designed to surface real loss: the agent must, at the end of each session,
report an exact 32-character ``DEPLOY_TOKEN`` from a tool result it fetched several
steps earlier. Under aggressive compression that token gets summarized away, so the
model re-calls the tool -- a re-request the LossDetector attributes as a loss event.
The policy then protects that tool and promotes its schema, so in later sessions the
token survives compression and the agent stops re-requesting. Loss falls over sessions,
this time from real API traffic.

Run (needs ANTHROPIC_API_KEY in env or .env):  python benchmarks/live_agent_loop.py
"""

from __future__ import annotations

import csv
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).parent.parent
PORT = 8123
BASE_URL = f"http://127.0.0.1:{PORT}"
MODEL = "claude-haiku-4-5-20251001"
N_SESSIONS = 10
MAX_STEPS = 8
DEPLOY_TOKEN = "a7f3c9e2b1d84f6099aa12ce77bb3401"  # the exact needle the agent must recall

# Aggressive compression + fast learning so the effect is visible in a few sessions.
# Set before importing contextos.main (it reads these at import time).
os.environ.setdefault("CONTEXTOS_COLDSTART_TIER1", "1")
os.environ.setdefault("CONTEXTOS_COLDSTART_TIER2", "2")
os.environ.setdefault("CONTEXTOS_SCHEMA_PROMOTE_THRESHOLD", "2")
os.environ.setdefault("CONTEXTOS_FIDELITY_STORE_PATH", str(ROOT / ".contextos" / "live_agent_fidelity.db"))
os.environ.setdefault("CONTEXTOS_POLICY_PATH", str(ROOT / ".contextos" / "live_agent_policy.json"))
os.environ.setdefault("CONTEXTOS_SCHEMA_PATH", str(ROOT / ".contextos" / "live_agent_schemas.json"))


# --- Local tools the agent can call (deterministic, no external calls). ------
_FILLER = "operational metadata that is verbose and not needed for the task " * 8


def _get_deploy_config() -> dict:
    return {
        "service": "api",
        "region": "us-east-1",
        "replicas": 3,
        "DEPLOY_TOKEN": DEPLOY_TOKEN,
        "notes": _FILLER,
    }


def _get_status() -> dict:
    return {"service": "api", "healthy": True, "uptime_s": 48213, "detail": _FILLER}


def _get_changelog() -> dict:
    return {"versions": [f"v1.{i}: {_FILLER}" for i in range(6)]}


TOOL_IMPLS = {
    "get_deploy_config": _get_deploy_config,
    "get_status": _get_status,
    "get_changelog": _get_changelog,
}

TOOLS = [
    {
        "name": name,
        "description": f"Return the {name.replace('_', ' ')}.",
        "input_schema": {"type": "object", "properties": {}},
    }
    for name in TOOL_IMPLS
]

INSTRUCTION = (
    "You are a deployment assistant. Do these steps strictly in order, one tool "
    "call at a time:\n"
    "1. Call get_deploy_config.\n"
    "2. Call get_status.\n"
    "3. Call get_changelog.\n"
    "4. Then reply with ONLY the exact DEPLOY_TOKEN value from the deploy config, "
    "character for character. If that token is not present in your current context, "
    "you MUST call get_deploy_config again to retrieve it before answering."
)


def _load_key() -> str | None:
    """Return the Anthropic key from env or a local .env file."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if key:
        return key
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip().startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _start_server() -> "uvicorn.Server":
    """Start the ContextOS server in a daemon thread and wait until it is ready."""
    import uvicorn

    import contextos.main as main  # imported here so env vars above take effect

    config = uvicorn.Config(main.app, host="127.0.0.1", port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()

    for _ in range(40):
        try:
            if httpx.get(f"{BASE_URL}/v1/loss-curve", timeout=1.0).status_code == 200:
                return server
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    raise RuntimeError("ContextOS server did not start in time")


def _run_session(client: object, session_id: str) -> dict:
    """Run one agentic session through ContextOS; return per-session metrics."""
    messages: list[dict] = [{"role": "user", "content": INSTRUCTION}]
    config_calls = 0
    final_text = ""

    for _ in range(MAX_STEPS):
        resp = client.messages.create(  # type: ignore[attr-defined]
            model=MODEL,
            max_tokens=400,
            tools=TOOLS,
            messages=messages,
            extra_headers={"contextos-session-id": session_id},
        )
        messages.append({"role": "assistant", "content": [b.model_dump() for b in resp.content]})

        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        if not tool_uses:
            final_text = "".join(b.text for b in resp.content if b.type == "text")
            break

        tool_results = []
        for tu in tool_uses:
            if tu.name == "get_deploy_config":
                config_calls += 1
            result = TOOL_IMPLS.get(tu.name, lambda: {"error": "unknown tool"})()
            tool_results.append(
                {"type": "tool_result", "tool_use_id": tu.id, "content": result}
            )
        messages.append({"role": "user", "content": tool_results})

    stats = httpx.get(f"{BASE_URL}/v1/stats/{session_id}", timeout=5.0).json()
    return {
        "session_id": session_id,
        "re_requested": config_calls > 1,        # a 2nd config fetch == lost context
        "config_calls": config_calls,
        "success": DEPLOY_TOKEN in final_text,
        "loss_events": stats["loss_events"],
        "compression_ratio": stats["compression_ratio"],
        "tokens_before": stats["tokens_before"],
        "tokens_after": stats["tokens_after"],
    }


def write_results(rows: list[dict]) -> Path:
    """Write the live curve CSV + append a summary; return the CSV path."""
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "live_loss_curve.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    early = rows[: max(1, len(rows) // 2)]
    late = rows[len(rows) // 2 :]
    early_loss = sum(r["re_requested"] for r in early) / len(early)
    late_loss = sum(r["re_requested"] for r in late) / len(late)
    avg_ratio = sum(r["compression_ratio"] for r in rows) / len(rows)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        f"ContextOS LIVE agent loop (real API traffic)  ({stamp})",
        "=" * 60,
        f"sessions                 : {len(rows)}",
        f"early loss rate (1st half): {early_loss:.0%}",
        f"late  loss rate (2nd half): {late_loss:.0%}",
        f"avg compression ratio     : {avg_ratio:.0%}",
        f"loss fell over sessions   : {'YES' if late_loss < early_loss else 'NO'}",
        "",
        "Per-session (re_request=loss | success=recalled token | comp ratio):",
    ]
    for r in rows:
        lines.append(
            f"  {r['session_id']}: re_request={int(r['re_requested'])} "
            f"success={int(r['success'])} loss_events={r['loss_events']} "
            f"comp={r['compression_ratio']:.2f}"
        )
    (results_dir / "live_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return csv_path


def main() -> int:
    """Boot the server, run the agent sessions, and write the real loss curve."""
    key = _load_key()
    if not key:
        print("No API key found (set ANTHROPIC_API_KEY or put it in .env).")
        return 1
    os.environ["ANTHROPIC_API_KEY"] = key  # server fallback + SDK

    server = _start_server()
    from anthropic import Anthropic

    client = Anthropic(api_key=key, base_url=BASE_URL)

    rows: list[dict] = []
    for s in range(N_SESSIONS):
        row = _run_session(client, session_id=f"live-{s}")
        rows.append(row)
        print(
            f"session {s}: re_request={int(row['re_requested'])} "
            f"success={int(row['success'])} loss_events={row['loss_events']} "
            f"comp={row['compression_ratio']:.2f}"
        )

    print()
    csv_path = write_results(rows)
    print(f"\nwrote {csv_path}")
    server.should_exit = True
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
