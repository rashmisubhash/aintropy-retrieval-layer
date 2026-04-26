"""Unit tests for scripts/benchmark_corpus.py helper functions.

Tests cover pure functions only — no model loading, no production data.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.benchmark_corpus as bc


# ----- load_production_corpus -----------------------------------------------

def test_load_production_corpus_returns_ids_and_texts(tmp_path):
    df = pd.DataFrame([
        {"doc_id": "d1", "text": "hello world"},
        {"doc_id": "d2", "text": "foo bar"},
    ])
    p = tmp_path / "corpus.parquet"
    df.to_parquet(p, index=False)

    ids, texts = bc.load_production_corpus(p)
    assert ids == ["d1", "d2"]
    assert texts == ["hello world", "foo bar"]


def test_load_production_corpus_preserves_order(tmp_path):
    rows = [{"doc_id": f"d{i}", "text": f"text {i}"} for i in range(20)]
    df = pd.DataFrame(rows)
    p = tmp_path / "corpus.parquet"
    df.to_parquet(p, index=False)

    ids, texts = bc.load_production_corpus(p)
    assert ids == [f"d{i}" for i in range(20)]
    assert texts == [f"text {i}" for i in range(20)]


# ----- load_production_queries ----------------------------------------------

def _write_queries(path: Path, entries: list[dict]) -> None:
    with open(path, "w") as f:
        json.dump(entries, f)


def test_load_production_queries_filters_missing_corpus_ids(tmp_path):
    p = tmp_path / "queries.json"
    _write_queries(p, [
        {"query_id": "q1", "query_text": "what is X", "relevant_doc_ids": ["d1", "d99"]},
        {"query_id": "q2", "query_text": "who is Y",  "relevant_doc_ids": ["d99"]},
    ])
    queries, qrels = bc.load_production_queries(p, {"d1", "d2"})

    assert "q1" in queries
    assert qrels["q1"] == {"d1": 1}  # d99 dropped — not in corpus
    assert "q2" not in queries       # all rel_ids missing → excluded entirely


def test_load_production_queries_all_grades_are_1(tmp_path):
    p = tmp_path / "queries.json"
    _write_queries(p, [
        {"query_id": "q1", "query_text": "foo", "relevant_doc_ids": ["d1", "d2"]},
    ])
    _, qrels = bc.load_production_queries(p, {"d1", "d2"})
    assert all(v == 1 for v in qrels["q1"].values())


def test_load_production_queries_empty_file(tmp_path):
    p = tmp_path / "queries.json"
    _write_queries(p, [])
    queries, qrels = bc.load_production_queries(p, {"d1"})
    assert queries == {}
    assert qrels == {}


# ----- vector_top_k ---------------------------------------------------------

def test_vector_top_k_returns_k_results():
    rng = np.random.default_rng(0)
    doc_embs = rng.standard_normal((50, 8)).astype(np.float32)
    doc_embs /= np.linalg.norm(doc_embs, axis=1, keepdims=True)
    query_emb = rng.standard_normal(8).astype(np.float32)
    query_emb /= np.linalg.norm(query_emb)

    results = bc.vector_top_k(query_emb, doc_embs, [f"d{i}" for i in range(50)], k=10)
    assert len(results) == 10


def test_vector_top_k_scores_descending():
    rng = np.random.default_rng(1)
    doc_embs = rng.standard_normal((100, 16)).astype(np.float32)
    doc_embs /= np.linalg.norm(doc_embs, axis=1, keepdims=True)
    query_emb = rng.standard_normal(16).astype(np.float32)
    query_emb /= np.linalg.norm(query_emb)

    results = bc.vector_top_k(query_emb, doc_embs, [f"d{i}" for i in range(100)], k=20)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_vector_top_k_finds_most_similar():
    dim = 8
    doc_embs = np.zeros((10, dim), dtype=np.float32)
    doc_embs[3][0] = 1.0  # only doc aligned with query
    query_emb = np.zeros(dim, dtype=np.float32)
    query_emb[0] = 1.0

    results = bc.vector_top_k(query_emb, doc_embs, [f"d{i}" for i in range(10)], k=1)
    assert results[0][0] == "d3"


# ----- normalise ------------------------------------------------------------

def test_normalise_produces_unit_vector():
    v = np.array([3.0, 4.0], dtype=np.float32)
    out = bc.normalise(v)
    assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-6


def test_normalise_zero_vector_passthrough():
    v = np.zeros(4, dtype=np.float32)
    out = bc.normalise(v)
    assert np.all(out == 0)


# ----- paraphrase generation ------------------------------------------------

def test_synonym_swap_replaces_known_word():
    rng_obj = __import__("random").Random(0)
    result = bc.synonym_swap("find the best method", rng_obj)
    assert result != "find the best method"


def test_synonym_swap_appends_explanation_on_no_match():
    rng_obj = __import__("random").Random(0)
    result = bc.synonym_swap("xyz zyx qrs", rng_obj)
    assert result.endswith("explanation")


def test_word_shuffle_swaps_within_same_words():
    rng_obj = __import__("random").Random(0)
    text = "one two three four five"
    result = bc.word_shuffle(text, rng_obj)
    assert set(result.split()) == set(text.split())
    assert result != text


def test_word_shuffle_short_text_prepends_prefix():
    rng_obj = __import__("random").Random(0)
    result = bc.word_shuffle("hi there", rng_obj)
    assert result.startswith("tell me about")


def test_make_paraphrases_returns_n_non_empty():
    rng_obj = __import__("random").Random(42)
    paras = bc.make_paraphrases("what is the best way to find a method", rng_obj, n=3)
    assert len(paras) == 3
    assert all(isinstance(p, str) and len(p) > 0 for p in paras)


# ----- aggregate_run --------------------------------------------------------

def test_aggregate_run_perfect_ranking():
    run = {"q1": {"d1": 10.0, "d2": -1.0, "d3": -2.0}}
    qrels = {"q1": {"d1": 1}}
    metrics = bc.aggregate_run(run, qrels)
    assert abs(metrics["ndcg@10"] - 1.0) < 1e-6
    assert abs(metrics["recall@10"] - 1.0) < 1e-6


def test_aggregate_run_relevant_doc_ranked_last():
    run = {"q1": {f"d{i}": float(100 - i) for i in range(99)}}
    run["q1"]["d_rel"] = -99.0  # scored last → rank 100
    qrels = {"q1": {"d_rel": 1}}
    metrics = bc.aggregate_run(run, qrels)
    assert metrics["ndcg@10"] == 0.0
    assert metrics["recall@10"] == 0.0
    assert metrics["recall@100"] == 1.0


def test_aggregate_run_averages_across_queries():
    run = {
        "q1": {"d1": 5.0, "d2": 1.0},
        "q2": {"d3": 1.0, "d4": 5.0},
    }
    qrels = {
        "q1": {"d1": 1},  # rank 1 → NDCG=1.0
        "q2": {"d3": 1},  # rank 2 → NDCG<1.0
    }
    metrics = bc.aggregate_run(run, qrels)
    assert 0.0 < metrics["ndcg@10"] < 1.0


# ----- percentile -----------------------------------------------------------

def test_percentile_median():
    assert bc.percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == pytest.approx(3.0)


def test_percentile_empty_returns_zero():
    assert bc.percentile([], 95) == 0.0
