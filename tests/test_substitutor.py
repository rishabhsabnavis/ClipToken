"""Unit tests for Module 3 -- SymbolSubstitutor.

Pure, deterministic module: no fixtures, mocks, or network needed.
"""

from __future__ import annotations

from contextos.modules.substitutor import SymbolSubstitutor


def test_build_table_multi_message_does_not_crash() -> None:
    """Regression: build_table over several messages must not raise.

    The original code called .split() on a list of contents, raising AttributeError.
    """
    long_word = "x" * 25
    messages = [{"content": long_word}, {"content": long_word}]
    table = SymbolSubstitutor().build_table(messages)
    assert table == {long_word: "$S1"}


def test_repeated_long_string_gets_symbol() -> None:
    """A long string seen >= min_occurrences earns a $S-style symbol."""
    long_word = "supercalifragilistic_token"
    messages = [{"content": long_word}, {"content": long_word}]
    table = SymbolSubstitutor().build_table(messages)
    assert table[long_word].startswith("$S")


def test_short_string_excluded() -> None:
    """Strings shorter than min_length never enter the table."""
    messages = [{"content": "short word"}, {"content": "short word"}]
    table = SymbolSubstitutor().build_table(messages, min_length=20)
    assert table == {}


def test_single_occurrence_excluded() -> None:
    """A long string seen only once is below min_occurrences and excluded."""
    long_word = "y" * 25
    messages = [{"content": long_word}]
    table = SymbolSubstitutor().build_table(messages, min_occurrences=2)
    assert table == {}


def test_ranking_by_length_times_count() -> None:
    """Higher (length * count) wins the lowest symbol number ($S1)."""
    big = "z" * 40          # length 40, count 2  -> score 80
    small = "w" * 20        # length 20, count 2  -> score 40
    messages = [
        {"content": f"{big} {small}"},
        {"content": f"{big} {small}"},
    ]
    table = SymbolSubstitutor().build_table(messages)
    assert table[big] == "$S1"
    assert table[small] == "$S2"


def test_substitute_replaces_key_with_symbol() -> None:
    """substitute swaps each table key for its symbol."""
    table = {"long_repeated_string_here": "$S1"}
    out = SymbolSubstitutor().substitute("see long_repeated_string_here now", table)
    assert out == "see $S1 now"


def test_substitute_longest_key_first() -> None:
    """Overlapping keys: the longer key is replaced wholesale, not clobbered."""
    table = {"foobar_baz_qux_token": "$S1", "foobar_baz": "$S2"}
    out = SymbolSubstitutor().substitute("foobar_baz_qux_token", table)
    assert out == "$S1"
