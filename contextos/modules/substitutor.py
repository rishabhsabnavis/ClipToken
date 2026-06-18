"""Module 3 -- SymbolSubstitutor.

Replaces recurring long strings with short symbols ($S1, $S2, ...) and keeps a legend
so nothing is lost. Pure Python, no external dependencies -- the easiest module to build
first.
"""

from __future__ import annotations
from collections import Counter
class SymbolSubstitutor:
    """Find long, frequently-repeated strings and swap them for short symbols."""

    def build_table(
        self,
        messages: list[dict],
        min_length: int = 20,
        min_occurrences: int = 2,
    ) -> dict:
        """Scan all message content and build a {long_string: symbol} table.

        Args:
            messages: the conversation messages to scan.
            min_length: ignore strings shorter than this (not worth a symbol).
            min_occurrences: only substitute strings seen at least this many times.

        Returns:
            A dict mapping each qualifying long string to a symbol like "$S1".
        """
        # Step 1: Flatten message content into one searchable corpus of strings.
        corpus = [msg['content'] for msg in messages]
        corpus = ' '.join(corpus).split(' ')
        corpus = [word for word in corpus if len(word) >= min_length]
        corpus_counter = Counter(corpus)
        corpus_counter = {word: count for word, count in corpus_counter.items() if count >= min_occurrences}
        corpus_counter = sorted(corpus_counter.items(), key=lambda x: len(x[0]) * x[1], reverse=True)
        corpus_counter = {word: f"$S{i+1}" for i, (word, count) in enumerate(corpus_counter)}
        return corpus_counter

    def substitute(self, text: str, table: dict) -> str:
        """Replace every table key in `text` with its symbol.

        Args:
            text: the text to rewrite.
            table: the {long_string: symbol} table from build_table.

        Returns:
            The text with long strings replaced by symbols.
        """
        for word, symbol in sorted(table.items(), key=lambda kv: len(kv[0]), reverse=True):
            text = text.replace(word, symbol)
        return text
