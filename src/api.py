"""FastAPI entrypoint for the retrieval middleware."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from src.retrieval import RetrievalEngine

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("retrieval-api")

_engine: RetrievalEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    _engine = RetrievalEngine()
    yield
    _engine = None


app = FastAPI(lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str


@app.get("/health")
def health() -> dict[str, Any]:
    assert _engine is not None
    return {"status": "ok", "corpus_size": _engine.collection.count()}


@app.post("/query")
def query(req: QueryRequest) -> dict[str, Any]:
    assert _engine is not None
    result = _engine.query(req.query)
    logger.info(
        json.dumps(
            {
                "event": "query",
                "query": req.query,
                "cache_hit": result["cache_hit"],
                "cache_score": result.get("cache_score"),
                "reranker_skipped": result.get("reranker_skipped", False),
                "reason": result.get("reason"),
                "num_results": len(result["results"]),
                "timings_ms": {
                    t["stage_name"]: round(t["duration_ms"], 3)
                    for t in result["timings"]
                },
            }
        )
    )
    return result


@app.post("/cache/clear")
def cache_clear() -> dict[str, Any]:
    """Clear the semantic cache. Admin endpoint for benchmarking."""
    assert _engine is not None
    _engine.cache.clear()
    return {"status": "cleared"}


@app.get("/cache/stats")
def cache_stats() -> dict[str, Any]:
    """Return cache occupancy + hit-rate counters."""
    assert _engine is not None
    return _engine.cache.cache_stats()
