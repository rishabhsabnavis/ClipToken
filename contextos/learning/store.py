"""Layer A -- FidelityStore (loss is reversible).

Every original chunk is written here *before* it is compressed, keyed by the
sha256 of its content. The wire payload only ever carries the compressed form
(plus a ``⟨ref:hash⟩`` marker for dropped segments), but the full original is
always recoverable from this store on demand.

This is what lets ContextOS describe itself as "compressed in transit, full
fidelity recoverable" instead of "lossy": on-the-wire loss becomes deferred
detail rather than permanent loss.

Backend: SQLite on disk for the MVP (single content-addressed table). Redis is a
later upgrade. The DB path comes from ``CONTEXTOS_FIDELITY_STORE_PATH``.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class FidelityStore:
    """Content-addressed store of original chunks, keyed by sha256(content)."""

    def __init__(self, db_path: str) -> None:
        """Open (or create) the SQLite store at db_path.

        Args:
            db_path: filesystem path to the SQLite file
                (``CONTEXTOS_FIDELITY_STORE_PATH``, e.g. ``./.contextos/fidelity.db``).
        """


        self.db_path = db_path


        Path(db_path).parent.mkdir(parents=True, exist_ok=True)


        # check_same_thread=False: FastAPI serves requests from a threadpool, so the
        # connection created here (import thread) is reused across worker threads.
        self.conn = sqlite3.connect(db_path, check_same_thread=False)


        #         (columns: hash TEXT PRIMARY KEY, content TEXT).
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS fidelity ("
            "hash TEXT PRIMARY KEY, "
            "content TEXT NOT NULL)"
        )
        self.conn.commit()
        logger.info("FidelityStore opened at db_path=%s", db_path)



    def put(self, content: str) -> str:
        """Store an original chunk and return its content hash.

        Storing is idempotent: the same content always maps to the same hash, so
        re-storing an existing chunk is a no-op that returns the same key.

        Args:
            content: the full, original (uncompressed) chunk.

        Returns:
            The sha256 hex digest of content, used as the lookup key.
        """
        # Step 1: Compute content_hash = sha256(content) hex digest.
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        # Step 2: INSERT OR IGNORE (hash, content) into the table.
        #         IGNORE makes re-storing the same content a harmless no-op.
        self.conn.execute(
            "INSERT OR IGNORE INTO fidelity (hash, content) VALUES (?, ?)",
            (content_hash, content),
        )
        # Step 3: Commit; log a put with the hash and content length.
        self.conn.commit()
        logger.info("put hash=%s content_len=%d", content_hash, len(content))
        # Step 4: Return content_hash.
        return content_hash



    def get(self, content_hash: str) -> str:
        """Restore the original chunk for a given content hash.

        Args:
            content_hash: a key previously returned by put().

        Returns:
            The original content string.

        Raises:
            KeyError: if no chunk is stored under content_hash.
        """
        # Step 1: SELECT content WHERE hash = content_hash.
        cursor = self.conn.execute(
            "SELECT content FROM fidelity WHERE hash = ?",
            (content_hash,),
        )
        row = cursor.fetchone()
        # Step 2: If no row, raise KeyError(content_hash).
        if row is None:
            raise KeyError(content_hash)
        # Step 3: Return the stored content.
        return row[0]
