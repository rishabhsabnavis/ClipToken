"""Layer B -- LossDetector (find the mistakes).

Detects evidence that a past compression discarded something the agent actually
needed. Three signals, cheapest first:

* **re-request** -- the agent re-runs a tool whose result was already compressed.
  This is free and a strong signal: if the compressed form had been enough, the
  agent would not be asking again.
* **shadow divergence** -- on a sampled fraction of turns, the caller also runs the
  raw (uncompressed) context; if the next action diverges, that is a loss event.
* **judge** -- an optional cheap Haiku call asking whether the compressed context
  dropped anything that changed the answer.

Each :class:`LossEvent` records the offending ``segment_type``, ``tool_name`` and the
compression ``level`` that caused it, so the CompressionPolicy can protect that
combination next time.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any

from contextos.schemas.models import LossEvent

if TYPE_CHECKING:
    from anthropic import Anthropic

    from contextos.schemas.models import Session

logger = logging.getLogger(__name__)


class LossDetector:
    """Watch responses for evidence that a compression discarded needed context."""

    @staticmethod
    def _args_key(args: Any) -> str:
        """Stable string key for a tool call's arguments (order-independent)."""
        import json

        return json.dumps(args, sort_keys=True, default=str)

    def __init__(
        self,
        client: Anthropic | None = None,
        model: str | None = None,
        shadow_sample_rate: float = 0.0,
        judge_enabled: bool = False,
    ) -> None:
        """Configure which loss signals are active.

        Args:
            client: Anthropic client, only needed for the (optional) judge signal.
            model: Haiku model id for the judge signal.
            shadow_sample_rate: fraction of scans that also compare a raw-context
                response (``CONTEXTOS_SHADOW_SAMPLE_RATE``); 0 disables shadowing.
            judge_enabled: whether to make the optional Haiku judge call.
        """
        self.client = client
        self.model = model
        self.shadow_sample_rate = shadow_sample_rate
        self.judge_enabled = judge_enabled
        # session_id -> tool_name -> {"info": {...}, "arg_keys": set[str]}.
        # Records which tool *calls* (name + arguments) had their results compressed
        # this session, so only a re-run of the SAME call -- not a different call to
        # the same tool -- is attributed as loss. "*" means "any arguments".
        self._compressed: dict[str, dict[str, dict[str, Any]]] = {}

    def note_compression(
        self,
        session_id: str,
        *,
        tool_name: str,
        level: str,
        segment_type: str = "tool_result",
        turn_index: int = -1,
        args: Any | None = None,
    ) -> None:
        """Record that a tool call's result was compressed, so a re-run can be blamed on it.

        Args:
            session_id: session the compression happened in.
            tool_name: the tool whose result was compressed.
            level: the compression level applied to it.
            segment_type: segment type for the resulting LossEvent.
            turn_index: index of the compressed turn (for the LossEvent).
            args: the arguments of the compressed call; None records "any arguments"
                (used by callers that do not track args, e.g. the simulator).
        """
        entry = self._compressed.setdefault(session_id, {}).setdefault(
            tool_name, {"info": {}, "arg_keys": set()}
        )
        entry["info"] = {
            "level": level,
            "segment_type": segment_type,
            "turn_index": turn_index,
        }
        entry["arg_keys"].add("*" if args is None else self._args_key(args))

    def scan(
        self,
        session: Session,
        response: dict,
        raw_response: dict | None = None,
    ) -> list[LossEvent]:
        """Return loss events implied by the model's response.

        Args:
            session: the session whose context was compressed.
            response: the provider response to the compressed context.
            raw_response: optional response to the *raw* context, used for the
                shadow-divergence signal when sampled.

        Returns:
            A list of detected LossEvents (possibly empty).
        """
        events: list[LossEvent] = []

        # --- Signal 1: re-request (free, strong). ---------------------------
        compressed = self._compressed.get(session.session_id, {})
        for tool_name, args in self._tool_calls(response):
            entry = compressed.get(tool_name)
            if entry is None:
                continue
            # Only a re-run of the *same* call counts (or a call the record marked
            # as matching any arguments).
            if "*" not in entry["arg_keys"] and self._args_key(args) not in entry["arg_keys"]:
                continue
            info = entry["info"]
            events.append(
                LossEvent(
                    segment_type=info["segment_type"],
                    tool_name=tool_name,
                    level=info["level"],
                    signal="re_request",
                    turn_index=info["turn_index"],
                )
            )
            logger.info(
                "loss re_request tool=%s level=%s session=%s",
                tool_name,
                info["level"],
                session.session_id,
            )

        # --- Signal 2: shadow divergence (sampled). -------------------------
        if raw_response is not None and random.random() < self.shadow_sample_rate:
            if self._diverges(response, raw_response):
                events.append(
                    LossEvent(
                        segment_type="assistant_reasoning",
                        tool_name=None,
                        level="unknown",
                        signal="shadow_divergence",
                        turn_index=-1,
                    )
                )
                logger.info("loss shadow_divergence session=%s", session.session_id)

        # --- Signal 3: judge (optional Haiku call). -------------------------
        if self.judge_enabled and self.client is not None and self._judge_flags(response):
            events.append(
                LossEvent(
                    segment_type="assistant_reasoning",
                    tool_name=None,
                    level="unknown",
                    signal="judge",
                    turn_index=-1,
                )
            )
            logger.info("loss judge session=%s", session.session_id)

        return events

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tool_calls(response: dict) -> list[tuple[str, Any]]:
        """Extract the (tool name, arguments) pairs the model asked to call."""
        calls: list[tuple[str, Any]] = []
        for block in response.get("content", []) or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                name = block.get("name")
                if name:
                    calls.append((name, block.get("input", {})))
        return calls

    @staticmethod
    def _diverges(response: dict, raw_response: dict) -> bool:
        """Whether the compressed-context action differs from the raw-context action.

        Compares the sequence of (tool name + input) requested by each response; any
        difference counts as divergence for the MVP.
        """
        def actions(resp: dict) -> list[tuple]:
            out: list[tuple] = []
            for block in resp.get("content", []) or []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    out.append((block.get("name"), str(block.get("input"))))
            return out

        return actions(response) != actions(raw_response)

    def _judge_flags(self, response: dict) -> bool:
        """Ask Haiku whether the compressed context looks like it lost needed detail.

        Returns False on any error so the judge signal can never break a request.
        """
        try:
            text = " ".join(
                block.get("text", "")
                for block in response.get("content", []) or []
                if isinstance(block, dict) and block.get("type") == "text"
            )
            if not text.strip():
                return False
            prompt = (
                "Did the assistant's reply show it was missing context (asking for "
                "info it should already have)? Answer yes or no.\n" + text[:400]
            )
            result = self.client.messages.create(
                model=self.model,
                max_tokens=5,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = "".join(
                b.text for b in result.content if getattr(b, "type", None) == "text"
            )
            return answer.strip().lower().startswith("yes")
        except Exception:  # noqa: BLE001 -- the judge must never break a request
            logger.warning("judge signal failed; skipping", exc_info=True)
            return False
