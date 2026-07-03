"""Request-path orchestration for ContextOS.

The :class:`Pipeline` wires Modules 1-5 into a single ``process`` call that turns an
incoming Anthropic-style request into a compressed messages list ready for the
provider. It is deliberately dependency-injected: the caller constructs the (possibly
heavy) module instances once and hands them in, so the pipeline itself stays cheap to
create and easy to test.

Flow per call::

    raw messages
      -> Module 1  compress each tool result            -> {tool_use_id: compressed}
      -> build Turn history from the messages
      -> Module 2  drop near-duplicate turns            (optional; needs an embedder)
      -> Module 3  build symbol table + substitute
      -> Module 4  adaptive per-turn compression        (consults CompressionPolicy)
      -> Module 5  assemble final messages + legend
      <- compressed messages list

The learning loop (FidelityStore, LossDetector, CompressionPolicy) is not on this hot
path except where Module 4 consults the policy and writes originals to the store; the
detector runs alongside via :meth:`observe`, kept separate so a call can compress even
before the detector exists.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from tiktoken import get_encoding

from contextos.schemas.models import Turn

if TYPE_CHECKING:
    from contextos.learning.policy import CompressionPolicy
    from contextos.modules.adaptive import AdaptiveCompressor
    from contextos.modules.assembler import ContextAssembler
    from contextos.modules.compressor import ToolResultCompressor
    from contextos.modules.deduplicator import SemanticDeduplicator
    from contextos.modules.substitutor import SymbolSubstitutor
    from contextos.schemas.models import MessagesRequest, Session

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrate the request-path modules for a single session."""

    def __init__(
        self,
        compressor: ToolResultCompressor,
        substitutor: SymbolSubstitutor,
        adaptive: AdaptiveCompressor,
        assembler: ContextAssembler,
        policy: CompressionPolicy,
        deduplicator: SemanticDeduplicator | None = None,
        *,
        dedup_threshold: float = 0.92,
        symbol_min_length: int = 20,
        symbol_min_occurrences: int = 2,
    ) -> None:
        """Wire the request-path modules and per-module config.

        Args:
            compressor: Module 1, compresses individual tool results.
            substitutor: Module 3, builds and applies the symbol table.
            adaptive: Module 4, per-turn adaptive compression (uses the policy + store).
            assembler: Module 5, produces the final messages list.
            policy: the CompressionPolicy consulted by the adaptive compressor.
            deduplicator: Module 2, optional; skipped when None (no embedder loaded).
            dedup_threshold: cosine similarity above which turns count as duplicates.
            symbol_min_length: minimum string length worth a symbol.
            symbol_min_occurrences: minimum repetitions before substituting a string.
        """
        self.compressor = compressor
        self.substitutor = substitutor
        self.adaptive = adaptive
        self.assembler = assembler
        self.policy = policy
        self.deduplicator = deduplicator
        self.dedup_threshold = dedup_threshold
        self.symbol_min_length = symbol_min_length
        self.symbol_min_occurrences = symbol_min_occurrences
        # Approximate token counts for before/after logging (see the other modules).
        self.encoding = get_encoding("cl100k_base")

    def process(self, request: MessagesRequest, session: Session) -> list[dict]:
        """Compress one request's messages and record stats on the session.

        Args:
            request: the incoming Anthropic-style request.
            session: per-session state; updated in place with token counts,
                turn history, and which modules fired.

        Returns:
            The compressed messages list to forward to the provider.
        """
        messages = request.messages
        modules_fired: list[str] = []

        # Count tokens on the *raw* messages so the ratio reflects real savings.
        tokens_before = self._count_messages(messages)

        # --- Module 1: compress every tool result to a short string. --------
        tool_results = self._compress_tool_results(messages)
        if tool_results:
            modules_fired.append("compressor")

        # --- Build Turn history from the raw messages. ----------------------
        # tool_result blocks are flattened to their already-compressed string so
        # the (lossy) adaptive step and token counts operate on the compressed
        # form, not the raw result.
        turns = self._build_turns(messages, tool_results)

        # --- Module 2: drop near-duplicate turns (optional). ----------------
        if self.deduplicator is not None and len(turns) > 1:
            turns = self._deduplicate_turns(turns)
            modules_fired.append("dedup")

        # --- Module 3: build the symbol table and substitute into turns. ----
        # Feed the substitutor the flattened turn text (it expects string content),
        # not the raw block messages.
        symbol_table = self.substitutor.build_table(
            [{"content": turn.content} for turn in turns],
            min_length=self.symbol_min_length,
            min_occurrences=self.symbol_min_occurrences,
        )
        if symbol_table:
            for turn in turns:
                turn.content = self.substitutor.substitute(turn.content, symbol_table)
            modules_fired.append("symbols")

        # --- Module 4: adaptive per-turn compression (consults the policy). -
        turns = self.adaptive.compress_history(turns, self.policy)
        modules_fired.append("adaptive")

        # --- Module 5: assemble the final wire messages. --------------------
        assembled = self.assembler.assemble(turns, symbol_table, tool_results)

        tokens_after = self._count_messages(assembled)

        # --- Record stats on the session. -----------------------------------
        session.turns = turns
        session.tokens_before += tokens_before
        session.tokens_after += tokens_after
        session.modules_fired = modules_fired

        logger.info(
            "pipeline session=%s tokens_before=%d tokens_after=%d ratio=%.3f modules=%s",
            session.session_id,
            tokens_before,
            tokens_after,
            self._ratio(tokens_before, tokens_after),
            ",".join(modules_fired),
        )
        return assembled

    # ------------------------------------------------------------------
    # Module 1 helpers -- tool result extraction / compression
    # ------------------------------------------------------------------

    def _compress_tool_results(self, messages: list[dict]) -> dict[str, str]:
        """Compress every tool_result block found in the messages.

        Args:
            messages: the raw request messages.

        Returns:
            A {tool_use_id: compressed_string} map for the assembler to inline.
        """
        # First map tool_use_id -> tool_name from assistant tool_use blocks so the
        # compressor can look up a schema for each result.
        names = self._tool_name_map(messages)
        compressed: dict[str, str] = {}
        for message in messages:
            for block in self._blocks(message):
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                tool_use_id = block.get("tool_use_id")
                if not tool_use_id:
                    continue
                raw = self._as_result_dict(block.get("content"))
                tool_name = names.get(tool_use_id, "unknown")
                compressed[tool_use_id] = self.compressor.compress(tool_name, raw)
        return compressed

    def _tool_name_map(self, messages: list[dict]) -> dict[str, str]:
        """Map each tool_use id to its tool name (from assistant tool_use blocks)."""
        names: dict[str, str] = {}
        for message in messages:
            for block in self._blocks(message):
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    if block.get("id"):
                        names[block["id"]] = block.get("name", "unknown")
        return names

    @staticmethod
    def _as_result_dict(content: Any) -> dict:
        """Coerce a tool_result's content into the dict the compressor expects."""
        if isinstance(content, dict):
            return content
        # Anthropic often sends a string or a list of blocks; wrap so the
        # compressor's json.dumps / summarization path has a dict to work with.
        return {"result": content}

    # ------------------------------------------------------------------
    # Turn construction / dedup
    # ------------------------------------------------------------------

    def _build_turns(
        self, messages: list[dict], tool_results: dict[str, str]
    ) -> list[Turn]:
        """Turn each raw message into a flat-text Turn (oldest first, 0-indexed).

        Each turn is tagged with (segment_type, tool_name) so the adaptive
        compressor keys its policy decision on the same combination the
        LossDetector attributes loss to -- this is what closes the learning loop.
        """
        names = self._tool_name_map(messages)
        turns: list[Turn] = []
        for index, message in enumerate(messages):
            segment_type, tool_name = self._classify_message(message, names)
            turns.append(
                Turn(
                    index=index,
                    role=message.get("role", "user"),
                    content=self._flatten_content(message, tool_results),
                    segment_type=segment_type,
                    tool_name=tool_name,
                )
            )
        return turns

    def _classify_message(
        self, message: dict, names: dict[str, str]
    ) -> tuple[str, str | None]:
        """Derive (segment_type, tool_name) from a raw message's blocks.

        A message carrying a tool_result is a "tool_result" segment tagged with the
        tool that produced it; otherwise it is classified by role.
        """
        for block in self._blocks(message):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                return "tool_result", names.get(block.get("tool_use_id"))
        if message.get("role") == "assistant":
            return "assistant_reasoning", None
        return "user", None

    def _deduplicate_turns(self, turns: list[Turn]) -> list[Turn]:
        """Drop near-duplicate turns via Module 2, then re-index the survivors."""
        contents = [turn.content for turn in turns]
        kept = set(
            self.deduplicator.deduplicate(contents, threshold=self.dedup_threshold)
        )
        # Keep the first turn per surviving content; drop later duplicates so a
        # string that appears twice collapses to a single turn.
        survivors: list[Turn] = []
        emitted: set[str] = set()
        for turn in turns:
            if turn.content in kept and turn.content not in emitted:
                survivors.append(turn)
                emitted.add(turn.content)
        # Re-index 0..n-1 so the adaptive compressor's age math stays correct.
        for new_index, turn in enumerate(survivors):
            turn.index = new_index
        return survivors

    def _flatten_content(self, message: dict, tool_results: dict[str, str]) -> str:
        """Flatten a message's content (string or block list) to plain text.

        tool_result blocks collapse to their Module-1 compressed string; tool_use
        blocks collapse to a short call marker.
        """
        content = message.get("content", "")
        if isinstance(content, str):
            return content

        parts: list[str] = []
        for block in self._blocks(message):
            if not isinstance(block, dict):
                parts.append(str(block))
            elif block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif block.get("type") == "tool_result":
                tool_use_id = block.get("tool_use_id", "")
                # Inline the compressed result; fall back to the raw content if,
                # for some reason, it was not compressed.
                parts.append(
                    tool_results.get(tool_use_id) or str(block.get("content", ""))
                )
            elif block.get("type") == "tool_use":
                parts.append(f"[call {block.get('name', 'tool')}]")
            else:
                parts.append(str(block.get("text", "")))
        return "\n".join(p for p in parts if p)

    @staticmethod
    def _blocks(message: dict) -> list:
        """Return a message's content blocks as a list (empty for plain strings)."""
        content = message.get("content", "")
        return content if isinstance(content, list) else []

    # ------------------------------------------------------------------
    # Token accounting
    # ------------------------------------------------------------------

    def _count_messages(self, messages: list[dict]) -> int:
        """Approximate total token count across all message content."""
        total = 0
        for message in messages:
            content = message.get("content", "")
            # Empty tool_results -> tool_result blocks fall back to their raw
            # content, so this counts the uncompressed size when given raw messages.
            text = (
                content
                if isinstance(content, str)
                else self._flatten_content(message, {})
            )
            total += len(self.encoding.encode(text))
        return total

    @staticmethod
    def _ratio(before: int, after: int) -> float:
        """Fraction of tokens saved (0.0 when there were none to begin with)."""
        return 1.0 - (after / before) if before else 0.0
