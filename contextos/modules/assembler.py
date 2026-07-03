"""Module 5 -- ContextAssembler.

Reassembles the final messages list to send to the LLM: prepends the symbol legend,
drops compressed tool results in place of the raw ones, and carries FidelityStore
reference markers for any dropped segments so they can be restored on demand.
"""

from __future__ import annotations

import logging

from contextos.schemas.models import Turn

logger = logging.getLogger(__name__)


class ContextAssembler:
    """Combine the outputs of Modules 1-4 into a final messages list."""

    def assemble(
        self,
        turns: list[Turn],
        symbol_table: dict,
        tool_results: dict,
    ) -> list[dict]:
        """Build the final messages payload for the LLM.

        Args:
            turns: compressed turn history from the AdaptiveCompressor.
            symbol_table: {long_string: symbol} table from the SymbolSubstitutor.
            tool_results: {tool_call_id: compressed_string} from the ToolResultCompressor.

        Returns:
            A messages list ready to forward to the provider.
        """
        messages: list[dict] = []

        # Step 1: If there are symbols, prepend a legend as a system-message addendum
        #         so the downstream model can decode $S1/$S2/... back to their strings.
        if symbol_table:
            messages.append(
                {"role": "system", "content": self._render_legend(symbol_table)}
            )

        # Step 2: Walk turns oldest-first, emitting one message each.
        for turn in turns:
            content = self._render_turn(turn, tool_results)
            messages.append({"role": turn.role, "content": content})

        logger.info(
            "assembled messages=%d symbols=%d tool_results=%d",
            len(messages),
            len(symbol_table),
            len(tool_results),
        )
        return messages

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _render_legend(self, symbol_table: dict) -> str:
        """Render the {long_string: symbol} table as a compact decode legend.

        Args:
            symbol_table: mapping of long string -> symbol (e.g. "$S1").

        Returns:
            A short human/model-readable legend, one "symbol = string" per line.
        """
        # Order by symbol number so the legend is stable and easy to scan.
        lines = [
            f"{symbol} = {long_string}"
            for long_string, symbol in sorted(
                symbol_table.items(), key=lambda kv: kv[1]
            )
        ]
        return "Symbol legend (expand these when reading):\n" + "\n".join(lines)

    def _render_turn(self, turn: Turn, tool_results: dict) -> str:
        """Pick the wire content for a single turn.

        Precedence:
            1. "drop" turns carry only a FidelityStore reference marker.
            2. Compressed turns use their summary.
            3. Verbatim turns use their original content.
        Any known tool_call_id appearing in the chosen content is then swapped for
        its compressed tool-result string.

        Args:
            turn: the (possibly compressed) turn to render.
            tool_results: {tool_call_id: compressed_string} replacements.

        Returns:
            The final content string for this turn.
        """
        # Step 1: dropped turns leave only a recoverable reference marker.
        if turn.level == "drop":
            return self._ref_marker(turn.fidelity_ref)

        # Step 2: compressed turns send their summary; verbatim send content.
        content = turn.summary if turn.summary is not None else turn.content

        # Step 3: substitute any raw tool-call id with its compressed result.
        for tool_call_id, compressed in tool_results.items():
            if tool_call_id in content:
                content = content.replace(tool_call_id, compressed)

        return content

    def _ref_marker(self, fidelity_ref: str | None) -> str:
        """Format the ⟨ref:hash⟩ marker used to stand in for a dropped segment.

        Args:
            fidelity_ref: the FidelityStore content hash, or None if missing.

        Returns:
            A ``⟨ref:hash⟩`` marker string (``⟨ref:missing⟩`` if the hash is absent).
        """
        return f"⟨ref:{fidelity_ref or 'missing'}⟩"
