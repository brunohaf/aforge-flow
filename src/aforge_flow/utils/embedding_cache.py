"""Persistent vector store for CLIP image embeddings.

Computing an image embedding is the only expensive step in the classifier
(a forward pass through a ViT on the GPU).  The *classification* itself is a
single matrix multiply against the cached text prototypes and is effectively
free.  So the whole throughput story reduces to: **embed every distinct
picture exactly once, ever, then never again.**

This module provides a tiny SQLite-backed key→vector store that survives
across runs with no external service and no extra dependencies (``sqlite3``
ships with Python).  Embeddings are stored as ``float16`` BLOBs, which halves
the on-disk size versus ``float32`` at no measurable accuracy cost for
zero-shot matching.

Keying strategy
---------------
The cache key is ``(model_key, content_key)`` where:

* ``model_key`` identifies the *encoder* (HF id + embedding dim).  Switching
  models therefore never returns stale vectors — each model gets its own
  namespace in the same DB.
* ``content_key`` identifies the *picture content*, supplied by the caller.
  Use a content hash (e.g. SHA-1 of the pixel bytes) or the perceptual hash
  already produced by the caption processor, so that re-encoded /
  re-resolution replicas of the same artwork share one embedding.

At 1152-d float16 each asset costs ~2.3 KB (10,000 assets ≈ 23 MB).
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class EmbeddingCache:
    """Thread-safe SQLite store mapping content keys to embedding vectors.

    The store is safe to share across the pipeline's worker threads: a single
    connection is guarded by a lock and opened with ``check_same_thread=False``.
    (For *multiprocessing*, give each process its own ``EmbeddingCache`` — SQLite
    handles concurrent readers and serialises writers via its own file lock.)
    """

    def __init__(self, db_path: Path, model_key: str, dim: int) -> None:
        """Open (creating if needed) the embedding store.

        Args:
            db_path: Path to the SQLite file. Parent dirs are created.
            model_key: Stable identifier for the encoder (e.g. the HF model id).
                Vectors from different ``model_key`` values never collide.
            dim: Embedding dimensionality, validated on read to catch a model
                swap that forgot to bump ``model_key``.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_key = model_key
        self.dim = dim
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                model_key   TEXT NOT NULL,
                content_key TEXT NOT NULL,
                vector      BLOB NOT NULL,
                PRIMARY KEY (model_key, content_key)
            )
            """)
        self._conn.commit()

    def get(self, content_key: str) -> np.ndarray | None:
        """Return the cached vector for *content_key*, or ``None`` if absent.

        Args:
            content_key: Content/perceptual hash identifying the picture.

        Returns:
            A ``float32`` 1-D array of length ``dim``, or ``None`` on a miss
            (or if a stored vector's length no longer matches ``dim``).
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT vector FROM embeddings WHERE model_key=? AND content_key=?",
                (self.model_key, content_key),
            ).fetchone()
        if row is None:
            return None
        vec = np.frombuffer(row[0], dtype=np.float16).astype(np.float32)
        if vec.shape[0] != self.dim:
            logger.warning(
                "embedding_dim_mismatch",
                content_key=content_key,
                expected=self.dim,
                found=int(vec.shape[0]),
            )
            return None
        return vec

    def get_many(self, content_keys: list[str]) -> dict[str, np.ndarray]:
        """Batch variant of :meth:`get` for a list of keys.

        Args:
            content_keys: Keys to look up.

        Returns:
            Dict containing only the keys that hit, mapped to ``float32`` vectors.
        """
        out: dict[str, np.ndarray] = {}
        if not content_keys:
            return out
        # Chunk to stay under SQLite's variable limit (999).
        with self._lock:
            for start in range(0, len(content_keys), 900):
                chunk = content_keys[start : start + 900]
                placeholders = ",".join("?" * len(chunk))
                rows = self._conn.execute(
                    f"SELECT content_key, vector FROM embeddings "  # nosec B608: only "?" placeholders
                    f"WHERE model_key=? AND content_key IN ({placeholders})",
                    (self.model_key, *chunk),
                ).fetchall()
                for key, blob in rows:
                    vec = np.frombuffer(blob, dtype=np.float16).astype(np.float32)
                    if vec.shape[0] == self.dim:
                        out[key] = vec
        return out

    def put(self, content_key: str, vector: np.ndarray) -> None:
        """Insert or replace the vector for *content_key*.

        Args:
            content_key: Content/perceptual hash identifying the picture.
            vector: 1-D embedding; stored as ``float16``.
        """
        self.put_many({content_key: vector})

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        """Bulk upsert vectors in a single transaction.

        Args:
            items: Mapping of content key to 1-D embedding vector.
        """
        if not items:
            return
        rows = [
            (self.model_key, key, np.asarray(vec, dtype=np.float16).tobytes())
            for key, vec in items.items()
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO embeddings (model_key, content_key, vector) "
                "VALUES (?, ?, ?)",
                rows,
            )
            self._conn.commit()

    def count(self) -> int:
        """Return the number of cached vectors for the active ``model_key``."""
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE model_key=?",
                (self.model_key,),
            ).fetchone()[0]

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()

    def __enter__(self) -> EmbeddingCache:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
