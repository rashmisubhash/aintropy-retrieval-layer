"""Three-condition retrieval benchmark on the 10k production corpus.

Conditions
----------
  1. cold      — vector search only (no cache, no reranker). Baseline.
  2. reranked  — vector search top-100 then cross-encoder rerank.
  3. warm      — semantic-cache + reranker pipeline. Cache is pre-warmed with
                 3 paraphrases of ~25% of eval queries, then all are evaluated.
                 Hit rate is reported separately.

Corpus / queries
----------------
  data/corpus.parquet   — 10k passages. Built by scripts/load_corpus.py.
  data/queries.json     — queries whose relevant doc is in the corpus.
                          Up to 200 are sampled deterministically for eval.

  Two qrel formats are supported (auto-detected):
    - Binary (MS MARCO dev/small): {"relevant_doc_ids": [...]}
      NDCG computed with grade=1 for every annotated relevant doc.
    - Graded (TREC DL 2019/2020): {"relevance_grades": {"pid": 0-3, ...}}
      NDCG computed with real grades — expect 0.65-0.75 range (honest).

Metrics
-------
  pytrec_eval at cutoffs @10, @20, @50, @100.
    - NDCG  : grade-weighted if graded qrels, binary otherwise.
    - MAP   : binary relevance (grade>=1 counts as relevant).
    - Recall: binary relevance.

Outputs
-------
  results/benchmark_corpus_results.json
  Console table comparing cold vs reranked + warm-cache stats.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytrec_eval
from sentence_transformers import CrossEncoder, SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cache import SemanticCache  # noqa: E402
from src.config import (  # noqa: E402
    CACHE_MAX_SIZE,
    CACHE_TTL_HOURS,
    CORPUS_VERSION,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    SEMANTIC_CACHE_THRESHOLD,
)

CORPUS_PATH = Path("data/corpus.parquet")
QUERIES_PATH = Path("data/queries.json")
RESULTS_PATH = Path("results/benchmark_corpus_results.json")

SEED = 42
N_EVAL_QUERIES = 200      # capped to available queries if fewer exist
N_PREWARM_QUERIES = 50    # capped to ~25% of eval_qids at runtime
PARAPHRASES_PER_QUERY = 3
VECTOR_TOP_K = 100
CUTOFFS = (10, 20, 50, 100)

NDCG_MEASURES = {f"ndcg_cut.{k}" for k in CUTOFFS}
BINARY_MEASURES = {f"map_cut.{k}" for k in CUTOFFS} | {f"recall.{k}" for k in CUTOFFS}


# ----- I/O ------------------------------------------------------------------

def load_production_corpus(path: Path) -> tuple[list[str], list[str]]:
    df = pd.read_parquet(path)
    return df["doc_id"].tolist(), df["text"].tolist()


def load_production_queries(
    path: Path,
    corpus_ids: set[str],
) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    with open(path) as f:
        raw = json.load(f)
    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    for entry in raw:
        qid = entry["query_id"]
        if "relevance_grades" in entry:
            # TREC DL graded mode: preserve grades 0-3 for honest NDCG.
            # Grade=0 entries that appear in top-k correctly lower NDCG.
            graded = {pid: g for pid, g in entry["relevance_grades"].items() if pid in corpus_ids}
            if not any(g >= 1 for g in graded.values()):
                continue
            queries[qid] = entry["query_text"]
            qrels[qid] = graded
        else:
            # MS MARCO binary mode
            rel_ids = [d for d in entry["relevant_doc_ids"] if d in corpus_ids]
            if not rel_ids:
                continue
            queries[qid] = entry["query_text"]
            qrels[qid] = {pid: 1 for pid in rel_ids}
    return queries, qrels


# ----- paraphrase generation ------------------------------------------------

SYNONYMS: dict[str, str] = {
    "best": "top",
    "biggest": "largest",
    "fastest": "quickest",
    "cheapest": "most affordable",
    "many": "multiple",
    "buy": "purchase",
    "make": "create",
    "use": "employ",
    "show": "display",
    "starts": "begins",
    "ends": "finishes",
    "find": "locate",
    "type": "kind",
    "types": "kinds",
    "begin": "start",
    "definition": "meaning",
    "kind": "type",
    "ways": "methods",
    "method": "way",
    "means": "ways",
    "help": "assist",
    "give": "provide",
    "get": "obtain",
}


def synonym_swap(text: str, rng: random.Random) -> str:
    words = text.split()
    candidates = [i for i, w in enumerate(words) if w.lower() in SYNONYMS]
    if not candidates:
        return text + " explanation"
    i = rng.choice(candidates)
    words[i] = SYNONYMS[words[i].lower()]
    return " ".join(words)


def word_shuffle(text: str, rng: random.Random) -> str:
    words = text.split()
    if len(words) < 4:
        return "tell me about " + text
    i = rng.randint(1, len(words) - 2)
    words[i], words[i + 1] = words[i + 1], words[i]
    return " ".join(words)


def make_paraphrases(text: str, rng: random.Random, n: int = 3) -> list[str]:
    out = [
        synonym_swap(text, rng),
        word_shuffle(text, rng),
        word_shuffle(synonym_swap(text, rng), rng),
    ]
    return out[:n]


# ----- retrieval primitives -------------------------------------------------

def vector_top_k(
    query_emb: np.ndarray, doc_embs: np.ndarray, doc_ids: list[str], k: int
) -> list[tuple[str, float]]:
    sims = doc_embs @ query_emb
    top_idx = np.argpartition(-sims, k - 1)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    return [(doc_ids[i], float(sims[i])) for i in top_idx]


def normalise(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return v.astype(np.float32, copy=False)
    return (v / n).astype(np.float32, copy=False)


# ----- metric aggregation ---------------------------------------------------

def aggregate_run(
    run: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
) -> dict[str, float]:
    ndcg_eval = pytrec_eval.RelevanceEvaluator(qrels, NDCG_MEASURES)
    binary_eval = pytrec_eval.RelevanceEvaluator(qrels, BINARY_MEASURES)
    ndcg_per_q = ndcg_eval.evaluate(run)
    bin_per_q = binary_eval.evaluate(run)

    def mean(measure_key: str, table: dict[str, dict[str, float]]) -> float:
        vals = [r.get(measure_key, 0.0) for r in table.values()]
        return float(np.mean(vals)) if vals else 0.0

    metrics: dict[str, float] = {}
    for k in CUTOFFS:
        metrics[f"ndcg@{k}"] = mean(f"ndcg_cut_{k}", ndcg_per_q)
        metrics[f"map@{k}"] = mean(f"map_cut_{k}", bin_per_q)
        metrics[f"recall@{k}"] = mean(f"recall_{k}", bin_per_q)
    return metrics


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


# ----- main -----------------------------------------------------------------

def main() -> None:
    rng = random.Random(SEED)

    print("Loading production corpus + queries...")
    doc_ids, doc_texts = load_production_corpus(CORPUS_PATH)
    corpus_id_set = set(doc_ids)
    queries, qrels = load_production_queries(QUERIES_PATH, corpus_id_set)
    print(f"  corpus: {len(doc_ids)} passages, eligible queries: {len(queries)}")

    all_qids = sorted(queries.keys())
    eval_qids = sorted(random.Random(SEED).sample(all_qids, min(N_EVAL_QUERIES, len(all_qids))))
    eval_queries = {qid: queries[qid] for qid in eval_qids}
    eval_qrels = {qid: qrels[qid] for qid in eval_qids}
    n_prewarm = max(5, min(N_PREWARM_QUERIES, len(eval_qids) // 4))
    print(f"  sampled {len(eval_qids)} queries for evaluation (prewarm: {n_prewarm})")

    print(f"\nLoading models: {EMBEDDING_MODEL}, {RERANKER_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    reranker = CrossEncoder(RERANKER_MODEL)
    embedder.encode(["warmup"], show_progress_bar=False)
    reranker.predict([["warmup query", "warmup doc"]])

    print(f"Embedding {len(doc_ids)} corpus passages...")
    doc_embs = embedder.encode(
        doc_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
    )
    doc_embs = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)
    text_by_id = dict(zip(doc_ids, doc_texts))

    # =====================================================================
    # Phase 1: cold — vector search only.
    # =====================================================================
    print("\n=== Phase 1: cold (vector search only) ===")
    cold_run: dict[str, dict[str, float]] = {}
    cold_lat: list[float] = []
    for qid in eval_qids:
        t0 = time.perf_counter_ns()
        emb = embedder.encode(
            [eval_queries[qid]], show_progress_bar=False, convert_to_numpy=True
        )[0]
        emb = normalise(emb)
        top = vector_top_k(emb, doc_embs, doc_ids, k=VECTOR_TOP_K)
        cold_lat.append((time.perf_counter_ns() - t0) / 1_000_000)
        cold_run[qid] = {pid: score for pid, score in top}
    cold_metrics = aggregate_run(cold_run, eval_qrels)
    cold_metrics["p95_latency_ms"] = percentile(cold_lat, 95)
    print(
        f"  done: NDCG@10={cold_metrics['ndcg@10']:.3f} "
        f"Recall@100={cold_metrics['recall@100']:.3f} "
        f"p95={cold_metrics['p95_latency_ms']:.0f}ms"
    )

    if cold_metrics["recall@100"] < 0.70:
        print(
            f"\n⚠️  Recall@100 = {cold_metrics['recall@100']:.2f} — many relevant docs "
            f"not in top-100 of 10k. This is expected to be lower than the gold-set benchmark."
        )

    # =====================================================================
    # Phase 2: reranked — vector top-100, cross-encoder rerank.
    # =====================================================================
    print("\n=== Phase 2: reranked (vector top-100 -> cross-encoder) ===")
    rerank_run: dict[str, dict[str, float]] = {}
    rerank_lat: list[float] = []
    for qid in eval_qids:
        t0 = time.perf_counter_ns()
        emb = embedder.encode(
            [eval_queries[qid]], show_progress_bar=False, convert_to_numpy=True
        )[0]
        emb = normalise(emb)
        top = vector_top_k(emb, doc_embs, doc_ids, k=VECTOR_TOP_K)
        pairs = [[eval_queries[qid], text_by_id[pid]] for pid, _ in top]
        scores = reranker.predict(pairs)
        ranked = sorted(
            ((pid, float(s)) for (pid, _), s in zip(top, scores)),
            key=lambda x: x[1],
            reverse=True,
        )
        rerank_lat.append((time.perf_counter_ns() - t0) / 1_000_000)
        rerank_run[qid] = {pid: score for pid, score in ranked}
    rerank_metrics = aggregate_run(rerank_run, eval_qrels)
    rerank_metrics["p95_latency_ms"] = percentile(rerank_lat, 95)
    print(
        f"  done: NDCG@10={rerank_metrics['ndcg@10']:.3f} "
        f"Recall@100={rerank_metrics['recall@100']:.3f} "
        f"p95={rerank_metrics['p95_latency_ms']:.0f}ms"
    )

    # =====================================================================
    # Phase 3: warm — semantic cache + reranker on miss.
    # =====================================================================
    print("\n=== Phase 3: warm (semantic cache + reranker) ===")
    cache = SemanticCache(
        max_size=CACHE_MAX_SIZE,
        threshold=SEMANTIC_CACHE_THRESHOLD,
        ttl_hours=CACHE_TTL_HOURS,
    )

    sampled_qids = rng.sample(eval_qids, n_prewarm)
    sampled_set = set(sampled_qids)
    n_cache_writes = 0
    for qid in sampled_qids:
        for para in make_paraphrases(eval_queries[qid], rng, n=PARAPHRASES_PER_QUERY):
            emb = embedder.encode(
                [para], show_progress_bar=False, convert_to_numpy=True
            )[0]
            emb = normalise(emb)
            cached, _ = cache.get(emb, corpus_version=CORPUS_VERSION)
            if cached is not None:
                continue
            top = vector_top_k(emb, doc_embs, doc_ids, k=VECTOR_TOP_K)
            pairs = [[para, text_by_id[pid]] for pid, _ in top]
            scores = reranker.predict(pairs)
            ranked = sorted(
                ((pid, float(s)) for (pid, _), s in zip(top, scores)),
                key=lambda x: x[1],
                reverse=True,
            )
            payload = {"results": [{"doc_id": pid, "score": s} for pid, s in ranked]}
            cache.set(emb, payload, corpus_version=CORPUS_VERSION)
            n_cache_writes += 1
    print(
        f"  pre-warmed cache: {n_prewarm} queries x "
        f"{PARAPHRASES_PER_QUERY} paraphrases -> {n_cache_writes} entries stored"
    )

    cache._lookups = 0
    cache._hits = 0
    cache._hit_similarities.clear()

    warm_run: dict[str, dict[str, float]] = {}
    warm_lat: list[float] = []
    warm_hit_lat: list[float] = []
    prewarm_hits = 0
    unseen_hits = 0
    for qid in eval_qids:
        t0 = time.perf_counter_ns()
        emb = embedder.encode(
            [eval_queries[qid]], show_progress_bar=False, convert_to_numpy=True
        )[0]
        emb = normalise(emb)
        cached, _sim = cache.get(emb, corpus_version=CORPUS_VERSION)
        if cached is not None:
            results = cached["results"]
            elapsed = (time.perf_counter_ns() - t0) / 1_000_000
            warm_lat.append(elapsed)
            warm_hit_lat.append(elapsed)
            if qid in sampled_set:
                prewarm_hits += 1
            else:
                unseen_hits += 1
        else:
            top = vector_top_k(emb, doc_embs, doc_ids, k=VECTOR_TOP_K)
            pairs = [[eval_queries[qid], text_by_id[pid]] for pid, _ in top]
            scores = reranker.predict(pairs)
            ranked = sorted(
                ((pid, float(s)) for (pid, _), s in zip(top, scores)),
                key=lambda x: x[1],
                reverse=True,
            )
            results = [{"doc_id": pid, "score": s} for pid, s in ranked]
            cache.set(emb, {"results": results}, corpus_version=CORPUS_VERSION)
            warm_lat.append((time.perf_counter_ns() - t0) / 1_000_000)
        warm_run[qid] = {r["doc_id"]: float(r["score"]) for r in results[:VECTOR_TOP_K]}
    warm_metrics = aggregate_run(warm_run, eval_qrels)
    warm_metrics["p95_latency_ms"] = percentile(warm_lat, 95)
    n_unseen = len(eval_qids) - n_prewarm
    warm_metrics["cache_hit_rate"] = (prewarm_hits + unseen_hits) / len(eval_qids)
    warm_metrics["cache_p95_latency_ms"] = percentile(warm_hit_lat, 95)
    prewarm_hit_rate = prewarm_hits / max(1, n_prewarm)
    unseen_hit_rate = unseen_hits / max(1, n_unseen)
    print(
        f"  done: NDCG@10={warm_metrics['ndcg@10']:.3f} "
        f"hit_rate={warm_metrics['cache_hit_rate']:.1%} "
        f"warm_p95={warm_metrics['p95_latency_ms']:.0f}ms "
        f"hit_p95={warm_metrics['cache_p95_latency_ms']:.0f}ms"
    )

    # =====================================================================
    # JSON output.
    # =====================================================================
    out = {
        "cold": cold_metrics,
        "reranked": rerank_metrics,
        "warm": warm_metrics,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {RESULTS_PATH}")

    # =====================================================================
    # Console table.
    # =====================================================================
    def pct_delta(a: float, b: float) -> str:
        if a == 0:
            return "n/a"
        return f"{(b - a) / a * 100:+.1f}%"

    print()
    print(f"{'Metric':<14}| {'Cold':<8}| {'Reranked':<9}| {'Delta':<14}")
    print("-" * 14 + "|" + "-" * 9 + "|" + "-" * 10 + "|" + "-" * 14)
    rows: list[tuple[str, str, str | None]] = [
        *[(f"NDCG@{k}", f"ndcg@{k}", None) for k in CUTOFFS],
        *[(f"MAP@{k}", f"map@{k}", None) for k in CUTOFFS],
        *[
            (
                f"Recall@{k}",
                f"recall@{k}",
                "target: 100%" if k == VECTOR_TOP_K else "—",
            )
            for k in CUTOFFS
        ],
    ]
    for label, key, override in rows:
        c = cold_metrics[key]
        r = rerank_metrics[key]
        delta = override if override else pct_delta(c, r)
        print(f"{label:<14}| {c:<8.3f}| {r:<9.3f}| {delta:<14}")
    cl = cold_metrics["p95_latency_ms"]
    rl = rerank_metrics["p95_latency_ms"]
    print(f"{'P95 Latency':<14}| {cl:<6.0f}ms | {rl:<7.0f}ms | {rl - cl:+.0f}ms")

    print()
    print("=== Warm cache ===")
    print(
        f"  cache hit rate (overall):       "
        f"{warm_metrics['cache_hit_rate']:.1%} "
        f"({prewarm_hits + unseen_hits}/{len(eval_qids)})"
    )
    print(
        f"  cache hit rate (pre-warmed {n_prewarm}): {prewarm_hit_rate:.1%} "
        f"({prewarm_hits}/{n_prewarm})"
    )
    print(
        f"  cache hit rate (unseen 150):    {unseen_hit_rate:.1%} "
        f"({unseen_hits}/{n_unseen})"
    )
    print(
        f"  warm p95 latency:               "
        f"{warm_metrics['p95_latency_ms']:.0f}ms"
    )
    print(
        f"  cache-hit p95 latency:          "
        f"{warm_metrics['cache_p95_latency_ms']:.0f}ms"
    )
    print(
        f"  warm NDCG@10 / Recall@100:      "
        f"{warm_metrics['ndcg@10']:.3f} / {warm_metrics['recall@100']:.3f}"
    )


if __name__ == "__main__":
    main()
