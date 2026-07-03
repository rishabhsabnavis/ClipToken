"""Real-API validation: run the fixture conversation through ContextOS against Anthropic.

This is the one step that upgrades the project from "validated in simulation" to
"validated on real traffic": it builds a *real* Anthropic client, compresses the
fixture conversation through the full request path, forwards the compressed call to
the live API, and reports the real token reduction and that the provider accepted the
compressed context and returned a usable answer.

The API key is read from the ``ANTHROPIC_API_KEY`` env var or a local ``.env`` file
(gitignored). The key is never printed. Uses a cheap model + small max_tokens to keep
the call to a few cents.

Run:  python eval/validate_live.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from contextos.learning.policy import CompressionPolicy
from contextos.learning.store import FidelityStore
from contextos.modules.adaptive import AdaptiveCompressor
from contextos.modules.assembler import ContextAssembler
from contextos.modules.compressor import ToolResultCompressor
from contextos.modules.substitutor import SymbolSubstitutor
from contextos.pipeline import Pipeline
from contextos.schemas.models import MessagesRequest, Session

ROOT = Path(__file__).parent.parent
FIXTURE = ROOT / "eval" / "fixtures" / "sample_conversation.json"
RESULTS = ROOT / "results" / "live_validation.txt"
CHEAP_MODEL = "claude-haiku-4-5-20251001"


def _load_key() -> str | None:
    """Return the Anthropic key from the environment or a local .env file."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if key:
        return key
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _log(line: str) -> None:
    """Append one timestamped line to the live-validation results file."""
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with RESULTS.open("a") as fh:
        fh.write(f"[{stamp}] {line}\n")


def main() -> int:
    """Run the live validation; return a process exit code."""
    key = _load_key()
    if not key:
        print(
            "No API key found.\n"
            "  Put it in a .env file at the project root (gitignored):\n"
            '    echo \'ANTHROPIC_API_KEY=sk-ant-...\' > .env\n'
            "  or export it in THIS shell before running.\n"
        )
        return 1

    from anthropic import Anthropic

    client = Anthropic(api_key=key)
    store = FidelityStore(str(ROOT / ".contextos" / "live_fidelity.db"))
    policy = CompressionPolicy(str(ROOT / ".contextos" / "live_policy.json"))
    pipe = Pipeline(
        compressor=ToolResultCompressor(client, CHEAP_MODEL),
        substitutor=SymbolSubstitutor(),
        adaptive=AdaptiveCompressor(
            client, CHEAP_MODEL, store, coldstart_tier1=1, coldstart_tier2=2
        ),
        assembler=ContextAssembler(),
        policy=policy,
        deduplicator=None,
    )

    conv = json.loads(FIXTURE.read_text())
    conv["model"] = CHEAP_MODEL      # keep the forward cheap
    conv["max_tokens"] = 300
    request = MessagesRequest(**conv)
    session = Session(session_id="live-validate")

    print("Compressing the fixture conversation (real Haiku summary calls)...")
    compressed = pipe.process(request, session)
    ratio = 1.0 - session.tokens_after / session.tokens_before
    print(
        f"  tokens_before={session.tokens_before}  tokens_after={session.tokens_after}"
        f"  compression_ratio={ratio:.1%}  modules={','.join(session.modules_fired)}"
    )

    # Forward the compressed call to the live API (lift the symbol legend into system).
    system_parts = [conv.get("system", "")]
    wire = []
    for msg in compressed:
        (system_parts.append(str(msg["content"])) if msg["role"] == "system"
         else wire.append(msg))
    system = "\n\n".join(p for p in system_parts if p) or None

    print("Forwarding compressed context to the live Anthropic API...")
    resp = client.messages.create(
        model=CHEAP_MODEL,
        max_tokens=300,
        system=system,
        messages=wire,
    )
    answer = "".join(b.text for b in resp.content if b.type == "text")
    ok = bool(answer.strip())
    print(f"  API status: {'OK' if ok else 'EMPTY'}  answer_chars={len(answer)}")
    print("  --- model answer (on compressed context) ---")
    print("  " + answer.strip()[:500].replace("\n", "\n  "))

    _log(
        f"live_validation compression_ratio={ratio:.3f} "
        f"tokens_before={session.tokens_before} tokens_after={session.tokens_after} "
        f"api_ok={ok} answer_chars={len(answer)}"
    )
    print(f"\nWrote {RESULTS}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
