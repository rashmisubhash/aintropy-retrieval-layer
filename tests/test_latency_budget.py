import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "chromadb" not in sys.modules:
    chromadb_stub = ModuleType("chromadb")

    class PersistentClient:
        def __init__(self, path):
            self.path = path

    chromadb_stub.PersistentClient = PersistentClient
    sys.modules["chromadb"] = chromadb_stub

if "sentence_transformers" not in sys.modules:
    sentence_transformers_stub = ModuleType("sentence_transformers")

    class CrossEncoder:
        def __init__(self, model_name):
            self.model_name = model_name

    class SentenceTransformer:
        def __init__(self, model_name):
            self.model_name = model_name

    sentence_transformers_stub.CrossEncoder = CrossEncoder
    sentence_transformers_stub.SentenceTransformer = SentenceTransformer
    sys.modules["sentence_transformers"] = sentence_transformers_stub

import src.retrieval as retrieval_module
from src.retrieval import RetrievalEngine


class FakeEmbedder:
    def encode(self, texts, show_progress_bar=False, convert_to_numpy=True):
        return np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32)


class FakeCollection:
    def query(self, query_embeddings, n_results, include):
        time.sleep(0.02)
        return {
            "ids": [[f"doc-{i}" for i in range(1, 7)]],
            "documents": [[f"document {i}" for i in range(1, 7)]],
        }


class FakeReranker:
    def __init__(self):
        self.called = False

    def predict(self, pairs):
        self.called = True
        raise AssertionError("reranker should be skipped when latency budget is exceeded")


class FakeCache:
    def __init__(self):
        self.set_called = False

    def get(self, embedding, corpus_version):
        return None, None

    def set(self, embedding, result, corpus_version):
        self.set_called = True


def test_rerank_skipped_when_vector_search_exceeds_budget(monkeypatch):
    monkeypatch.setattr(retrieval_module, "RERANK_SKIP_THRESHOLD_MS", 1)

    engine = RetrievalEngine.__new__(RetrievalEngine)
    engine.embedder = FakeEmbedder()
    engine.collection = FakeCollection()
    engine.reranker = FakeReranker()
    engine.cache = FakeCache()

    result = engine.query("slow query")

    assert result["reranker_skipped"] is True
    assert result["reason"] == "latency_budget"
    assert len(result["results"]) == 5
    assert [item["doc_id"] for item in result["results"]] == [
        "doc-1",
        "doc-2",
        "doc-3",
        "doc-4",
        "doc-5",
    ]
    assert all(item["score"] is None for item in result["results"])
    assert engine.reranker.called is False
    assert engine.cache.set_called is False
    assert [timing["stage_name"] for timing in result["timings"]] == [
        "embed",
        "cache_lookup",
        "vector_search",
    ]
