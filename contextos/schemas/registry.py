"""Tool name -> field list mappings used by Module 1 (ToolResultCompressor).

The registry can be *seeded* by hand (known tools) and *grown automatically* by the
learning loop: once the LossDetector + CompressionPolicy have observed which fields of
a tool's output the agent actually depends on, `learn_schema` promotes those fields
here, making that tool near-lossless from then on.
"""

from __future__ import annotations

import json
from pathlib import Path

# Seed schemas: tool_name -> list of fields worth keeping from its raw result.
# Start small; the learning loop fills this in over time.
schema_registry: dict[str, list[str]] = {
    # "web_search": ["title", "url", "snippet"],
    # "read_file":  ["path", "content"],
}


def get_schema(tool_name: str) -> list[str] | None:
    """Return the field list for a tool, or None if the tool is unknown.

    Args:
        tool_name: name of the tool whose result is being compressed.

    Returns:
        The list of fields to keep, or None to fall back to Haiku summarization.
    """
    # Step 1: Look up tool_name in schema_registry.
    # Step 2: Return the field list if present, else None.
    return schema_registry.get(tool_name)


def learn_schema(tool_name: str, fields: list[str]) -> None:
    """Record/merge the fields the agent has been observed to depend on for a tool.

    Called by CompressionPolicy.update when a tool's safe fields have been learned.

    Args:
        tool_name: tool to update.
        fields: fields observed to matter for downstream task success.
    """
    # Step 1: Merge `fields` into any existing entry for tool_name (union, keep order).
    merged = list(schema_registry.get(tool_name, []))
    for field_name in fields:
        if field_name not in merged:
            merged.append(field_name)
    # Step 2: Store back into schema_registry.
    schema_registry[tool_name] = merged


def save_registry(path: str) -> None:
    """Persist auto-learned schemas to disk so learning survives restarts.

    Args:
        path: JSON file to write the current schema_registry to.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(schema_registry, indent=2))


def load_registry(path: str) -> None:
    """Merge schemas previously persisted by save_registry into the live registry.

    Seeds already present are kept; loaded entries are merged field-wise so a restart
    resumes with everything the learning loop had promoted.

    Args:
        path: JSON file previously written by save_registry (ignored if absent).
    """
    file = Path(path)
    if not file.exists():
        return
    loaded: dict[str, list[str]] = json.loads(file.read_text())
    for tool_name, fields in loaded.items():
        learn_schema(tool_name, fields)
