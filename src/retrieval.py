"""Vector search + cross-encoder rerank, guarded by a single-tier semantic cache."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Optional

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.cache import SemanticCache
from src.config import (
    CACHE_MAX_SIZE,
    CACHE_TTL_HOURS,
    CHROMA_PATH,
    CORPUS_VERSION,
    EMBEDDING_MODEL,
    FINAL_TOP_K,
    LATENCY_BUDGET_MS,
    RERANKER_MODEL,
    RERANK_SKIP_THRESHOLD_MS,
    SEMANTIC_CACHE_THRESHOLD,
    VECTOR_TOP_K,
)
from src.timing import StageTimer, TimingResult

COLLECTION_NAME = "msmarco_passages"


class RetrievalEngine:
    """Loads models + Chroma collection once; serves queries with per-stage timings.

    Request flow per query:
      1. Embed the query (~15 ms — not the bottleneck).
      2. Semantic-cache probe with the embedding. Hit (similarity >= threshold,
         matching corpus_version, within TTL) -> return cached result with
         cache_score.
      3. Chroma kNN top VECTOR_TOP_K. If the request has already overrun
         RERANK_SKIP_THRESHOLD_MS, return the vector top FINAL_TOP_K directly
         and mark the response with reranker_skipped + reason="latency_budget".
      4. Cross-encoder rerank to top FINAL_TOP_K. Store the reranked payload
         in the cache (tagged with CORPUS_VERSION) and return.
    """

    def __init__(self) -> None:
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.reranker = CrossEncoder(RERANKER_MODEL)
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_collection(COLLECTION_NAME)
        self.cache = SemanticCache(
            max_size=CACHE_MAX_SIZE,
            threshold=SEMANTIC_CACHE_THRESHOLD,
            ttl_hours=CACHE_TTL_HOURS,
        )

    def query(self, text: str) -> dict[str, Any]:
        query_start_ns = time.perf_counter_ns()

        with StageTimer("embed") as t_embed:
            query_emb = self.embedder.encode(
                [text], show_progress_bar=False, convert_to_numpy=True
            )[0]

        with StageTimer("cache_lookup") as t_cache:
            cached, sim = self.cache.get(query_emb, corpus_version=CORPUS_VERSION)

        if cached is not None:
            return self._format(
                cached,
                timings=[t_embed.result, t_cache.result],
                cache_hit=True,
                cache_score=sim,
            )

        with StageTimer("vector_search") as t_search:
            raw = self.collection.query(
                query_embeddings=[query_emb.tolist()],
                n_results=VECTOR_TOP_K,
                include=["documents"],
            )
        doc_ids: list[str] = raw["ids"][0]
        docs: list[str] = raw["documents"][0]

        elapsed_ms = (time.perf_counter_ns() - query_start_ns) / 1_000_000
        if elapsed_ms > RERANK_SKIP_THRESHOLD_MS:
            results = [
                {"doc_id": did, "text": doc, "score": None}
                for did, doc in zip(doc_ids[:FINAL_TOP_K], docs[:FINAL_TOP_K])
            ]
            payload = {"results": results}
            return self._format(
                payload,
                timings=[t_embed.result, t_cache.result, t_search.result],
                cache_hit=False,
                cache_score=None,
                reranker_skipped=True,
                reason="latency_budget",
            )

        with StageTimer("rerank") as t_rerank:
            pairs = [[text, doc] for doc in docs]
            scores = self.reranker.predict(pairs)
            ranked = sorted(
                zip(doc_ids, docs, scores),
                key=lambda triple: float(triple[2]),
                reverse=True,
            )[:FINAL_TOP_K]
            results = [
                {"doc_id": did, "text": doc, "score": float(score)}
                for did, doc, score in ranked
            ]

        payload = {"results": results}
        self.cache.set(query_emb, payload, corpus_version=CORPUS_VERSION)

        return self._format(
            payload,
            timings=[
                t_embed.result,
                t_cache.result,
                t_search.result,
                t_rerank.result,
            ],
            cache_hit=False,
            cache_score=None,
        )

    @staticmethod
    def _format(
        payload: dict[str, Any],
        timings: list[TimingResult],
        *,
        cache_hit: bool,
        cache_score: Optional[float],
        reranker_skipped: bool = False,
        reason: Optional[str] = None,
    ) -> dict[str, Any]:
        return {
            "results": payload["results"],
            "timings": [asdict(t) for t in timings],
            "cache_hit": cache_hit,
            "cache_score": cache_score,
            "reranker_skipped": reranker_skipped,
            "reason": reason,
            "latency_budget_ms": LATENCY_BUDGET_MS,
        }
