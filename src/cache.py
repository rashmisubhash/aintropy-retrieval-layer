"""Single-tier semantic query cache.

Why one tier and not two
------------------------
An earlier version had a separate exact-string cache fronting the semantic
cache so literal repeats could skip embedding entirely. With our embedding
model (~15 ms per call) the saved cost on literal repeats is below most SLA
budgets, and the dual-cache code path doubled the surface area for eviction,
TTL, and consistency bugs. We collapse to a single semantic cache: every
query embeds first (~15 ms), then a cosine NN scan over cached embeddings
decides hit/miss. A literal repeat is just the trivial case of similarity 1.0
above the 0.92 threshold.

Entries are dict-keyed on the SHA-256 of the embedding rounded to 4 decimal
places, which makes the key float-noise resilient and lets repeat-stores of
the same query overwrite cleanly. The cosine NN scan runs over the actual
(normalised) vectors stored alongside.

Each entry carries:
  - embedding (L2-normalised np.ndarray)
  - result payload
  - corpus_version: a tag identifying which corpus produced the answer
  - timestamp: wall-clock seconds at insertion

Lookup honours both:
  - corpus_version mismatch -> miss (lets corpus refreshes invalidate cleanly)
  - timestamp older than TTL -> miss
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class CacheEntry:
    embedding: np.ndarray  # L2-normalised
    result: dict[str, Any]
    corpus_version: str
    timestamp: float  # epoch seconds


class SemanticCache:
    """Single-tier semantic cache. See module docstring for rationale."""

    def __init__(
        self, max_size: int, threshold: float, ttl_hours: float
    ) -> None:
        self.max_size = max_size
        self.threshold = threshold
        self.ttl_seconds = ttl_hours * 3600.0
        self._entries: dict[str, CacheEntry] = {}
        # Stats counters.
        self._lookups = 0
        self._hits = 0
        self._hit_similarities: list[float] = []

    @staticmethod
    def _hash_key(embedding: np.ndarray) -> str:
        rounded = np.round(embedding.astype(np.float64), 4)
        return hashlib.sha256(rounded.tobytes()).hexdigest()

    @staticmethod
    def _normalise(v: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            return v.astype(np.float32, copy=False)
        return (v / norm).astype(np.float32, copy=False)

    def get(
        self, embedding: np.ndarray, corpus_version: str
    ) -> tuple[Optional[dict[str, Any]], Optional[float]]:
        """Return (result, similarity) on hit, (None, None) on miss.

        A hit requires:
          (1) at least one cached entry with the same corpus_version,
          (2) within TTL,
          (3) cosine similarity to the query embedding >= threshold.
        """
        self._lookups += 1
        if not self._entries:
            return None, None

        now = time.time()
        valid_keys: list[str] = []
        valid_embs: list[np.ndarray] = []
        for key, entry in self._entries.items():
            if entry.corpus_version != corpus_version:
                continue
            if now - entry.timestamp > self.ttl_seconds:
                continue
            valid_keys.append(key)
            valid_embs.append(entry.embedding)

        if not valid_embs:
            return None, None

        q = self._normalise(embedding)
        mat = np.stack(valid_embs)
        sims = mat @ q
        idx = int(np.argmax(sims))
        best = float(sims[idx])

        if best >= self.threshold:
            self._hits += 1
            self._hit_similarities.append(best)
            return self._entries[valid_keys[idx]].result, best
        return None, None

    def set(
        self,
        embedding: np.ndarray,
        result: dict[str, Any],
        corpus_version: str,
    ) -> None:
        """Insert or overwrite an entry. Evicts oldest by timestamp at cap."""
        if len(self._entries) >= self.max_size:
            oldest_key = min(
                self._entries, key=lambda k: self._entries[k].timestamp
            )
            del self._entries[oldest_key]
        key = self._hash_key(embedding)
        self._entries[key] = CacheEntry(
            embedding=self._normalise(embedding),
            result=result,
            corpus_version=corpus_version,
            timestamp=time.time(),
        )

    def clear(self) -> None:
        self._entries.clear()
        self._lookups = 0
        self._hits = 0
        self._hit_similarities.clear()

    def cache_stats(self) -> dict[str, Any]:
        avg_sim = (
            float(np.mean(self._hit_similarities))
            if self._hit_similarities
            else 0.0
        )
        hit_rate = self._hits / self._lookups if self._lookups else 0.0
        return {
            "total_entries": len(self._entries),
            "lookups": self._lookups,
            "hits": self._hits,
            "hit_rate": hit_rate,
            "avg_hit_similarity": avg_sim,
        }

    def __len__(self) -> int:
        return len(self._entries)
