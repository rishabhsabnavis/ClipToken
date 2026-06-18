"""Module 1 -- ToolResultCompressor.

Turns a large, messy tool result into a short string (< 100 tokens). Uses a known
schema when available (lossless field extraction), otherwise asks Haiku to summarize.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anthropic import Anthropic


class ToolResultCompressor:
    """Compress a single tool result down to its useful fields or a short summary."""

    def __init__(self, client: Anthropic, model: str) -> None:
        """Store the Anthropic client and the Haiku model id used for fallback summaries.

        Args:
            client: Anthropic SDK client (used only when no schema matches).
            model: Haiku model id, e.g. "claude-haiku-4-5-20251001".
        """
        # Step 1: Save client and model on self for later use.
        raise NotImplementedError

    def compress(self, tool_name: str, raw_result: dict) -> str:
        """Compress one tool result to a string under ~100 tokens.

        Args:
            tool_name: the tool that produced raw_result (used for schema lookup).
            raw_result: the full, raw tool output.

        Returns:
            A compressed string representation, always < 100 tokens.
        """
        # Step 1: Count tokens of raw_result (tiktoken) -> tokens_before, for logging.
        # Step 2: Look up tool_name in schema_registry (schemas.registry.get_schema).
        # Step 3a: If a schema exists -> extract only those fields, format compactly.
        # Step 3b: If no schema -> call self._summarize(raw_result) via Haiku.
        # Step 4: Count tokens of the result -> tokens_after; log tool_name/before/after.
        # Step 5: Return the compressed string.
        raise NotImplementedError

    def _summarize(self, raw_result: dict) -> str:
        """Fallback: ask Haiku to summarize an unknown tool result in < 100 tokens.

        Args:
            raw_result: the raw tool output to summarize.

        Returns:
            A Haiku-generated summary string.
        """
        # Step 1: Build a compact prompt (keep prompt < 200 input tokens per conventions).
        # Step 2: Call self.client.messages.create with self.model.
        # Step 3: Extract and return the text content.
        raise NotImplementedError
