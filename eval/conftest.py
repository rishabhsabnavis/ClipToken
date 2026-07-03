"""Shared offline test fixtures.

There is no API key in the test environment, so every test runs against a fake
Anthropic client that returns small, deterministic responses. This lets us exercise
the full request path (which makes Haiku summary calls) and the FastAPI surface
without any network access.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "sample_conversation.json"


@dataclass
class _Block:
    """A minimal stand-in for an Anthropic content block."""

    type: str
    text: str


class _Message:
    """A minimal stand-in for an Anthropic message response."""

    def __init__(self, text: str) -> None:
        self.id = "msg_fake"
        self.type = "message"
        self.role = "assistant"
        self.model = "claude-fake"
        self.stop_reason = "end_turn"
        self.content = [_Block(type="text", text=text)]

    def model_dump(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "role": self.role,
            "model": self.model,
            "stop_reason": self.stop_reason,
            "content": [{"type": b.type, "text": b.text} for b in self.content],
        }


class _Messages:
    """Fake ``client.messages`` namespace."""

    def create(self, **kwargs: Any) -> _Message:
        # Size the fake summary by the caller's max_tokens so summaries stay small
        # and token reduction is real (bullet ~120, sentence ~60, tool ~100, judge 5).
        max_tokens = kwargs.get("max_tokens", 100)
        if max_tokens <= 5:
            return _Message("no")
        if max_tokens <= 60:
            return _Message("Key point retained for the agent's next step.")
        if max_tokens <= 120:
            return _Message("- fact one retained\n- fact two retained")
        return _Message("Compact summary of the tool result kept under budget.")


class FakeAnthropic:
    """Fake Anthropic client: ``.messages.create`` returns deterministic responses."""

    def __init__(self, *_: Any, **__: Any) -> None:
        self.messages = _Messages()


@pytest.fixture
def fake_client() -> FakeAnthropic:
    """A fresh fake Anthropic client."""
    return FakeAnthropic()


@pytest.fixture
def conversation() -> dict:
    """The 10-turn research-agent fixture conversation."""
    return json.loads(FIXTURE.read_text())
