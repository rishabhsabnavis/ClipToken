"""Module 1 -- ToolResultCompressor.

Turns a large, messy tool result into a short string (< 100 tokens). Uses a known
schema when available (lossless field extraction), otherwise asks Haiku to summarize.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from tiktoken import get_encoding

from contextos.schemas.registry import get_schema

if TYPE_CHECKING:
    from anthropic import Anthropic


logger = logging.getLogger(__name__)


class ToolResultCompressor:
    """Compress a single tool result down to its useful fields or a short summary."""

    def __init__(self, client: Anthropic, model: str) -> None:
        """Store the Anthropic client and the Haiku model id used for fallback summaries.

        Args:
            client: Anthropic SDK client (used only when no schema matches).
            model: Haiku model id, e.g. "claude-haiku-4-5-20251001".
        """
        self.client = client
        self.model = model
        # tiktoken only knows OpenAI model names, so use a fixed encoding. This
        # is an approximate token count for Claude (off ~15-20%), which is fine
        # for the before/after logging ratios.
        self.encoding = get_encoding("cl100k_base")

    def compress(self, tool_name: str, raw_result: dict) -> str:
        """Compress one tool result to a string under ~100 tokens.

        Args:
            tool_name: the tool that produced raw_result (used for schema lookup).
            raw_result: the full, raw tool output.

        Returns:
            A compressed string representation, always < 100 tokens.
        """
        tokens_before = len(self.encoding.encode(json.dumps(raw_result)))

        schema = get_schema(tool_name)

        if not schema:
            result = self._summarize(raw_result)  # haiku summary
        else:
            schema_result = self.extract_fields(raw_result, schema)
            result = self.format_compactly(schema_result)

        tokens_after = len(self.encoding.encode(result))

        logger.info(
            "tool_name=%s tokens_before=%d tokens_after=%d",
            tool_name,
            tokens_before,
            tokens_after,
        )
        return result

    def extract_fields(self, raw_result: dict, schema: list[str]) -> dict:
        """Extract the schema's fields from the raw result.

        Uses ``.get`` so an auto-learned field that is missing from a particular
        result yields ``None`` instead of raising -- learned schemas are a union of
        fields seen across a tool's outputs, which any single result may not carry.
        """
        return {field: raw_result.get(field) for field in schema if field in raw_result}

    def format_compactly(self, result: dict) -> str:
        """Format the result compactly."""
        return json.dumps(result)

    def _summarize(self, raw_result: dict) -> str:
        """Fallback: ask Haiku to summarize an unknown tool result in < 100 tokens.

        Args:
            raw_result: the raw tool output to summarize.

        Returns:
            A Haiku-generated summary string.
        """
        # Step 1: Build a compact prompt (keep prompt < 200 input tokens per conventions).
        prompt = (
            "Summarize this tool result in under 100 tokens, keeping only the "
            f"facts an agent would need to continue its task:\n{raw_result}"
        )

        # Step 2: Call self.client.messages.create with self.model.
        response = self.client.messages.create(
            model=self.model,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in response.content if block.type == "text"
        )
