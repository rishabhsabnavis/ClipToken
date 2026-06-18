"""Module 5 -- ContextAssembler.

Reassembles the final messages list to send to the LLM: prepends the symbol legend,
drops compressed tool results in place of the raw ones, and carries FidelityStore
reference markers for any dropped segments so they can be restored on demand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contextos.schemas.models import Turn


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
        # Step 1: If symbol_table is non-empty, render a compact legend and prepend it as
        #         a system-message addendum so symbols are interpretable.
        # Step 2: Walk turns in order, emitting each as a message:
        #         - use turn.summary when the turn was compressed, else turn.content
        #         - for "drop" turns, emit only the ⟨ref:hash⟩ marker (turn.fidelity_ref).
        # Step 3: Replace raw tool results with their compressed strings from tool_results.
        # Step 4: Return the assembled messages list.
        raise NotImplementedError
