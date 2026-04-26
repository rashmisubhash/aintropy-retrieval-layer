"""Download MS MARCO passages, build a 10k-passage corpus, and index with ChromaDB.

Supports two dataset modes (--dataset flag):

  msmarco-dev-small  (default)
    Relevant passages from MS MARCO dev/small + random distractors.
    Binary qrels (0/1). ~7k queries.

  trec-dl-2019 | trec-dl-2020
    All passages judged by NIST for TREC Deep Learning 2019 or 2020.
    Corpus priority: grade>=1 passages first, then grade=0 hard negatives,
    then random distractors to reach CORPUS_SIZE.
    Graded qrels (0-3) written to queries.json so benchmark can use
    grade-weighted NDCG instead of binary NDCG.
    43 queries (2019) / 54 queries (2020) — fewer but more realistic.

Idempotent:
  - Skips corpus/queries build if data/corpus.parquet and data/queries.json both exist.
  - Skips Chroma indexing if the collection already has the expected doc count.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import ir_datasets
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    CHROMA_PATH,
    CORPUS_PATH,
    CORPUS_SIZE,
    EMBEDDING_MODEL,
    QUERIES_PATH,
)

RANDOM_SEED = 42
EMBED_BATCH = 256
COLLECTION_NAME = "msmarco_passages"


# ---------------------------------------------------------------------------
# MS MARCO dev/small (binary relevance)
# ---------------------------------------------------------------------------

def build_corpus_and_queries_msmarco() -> tuple[pd.DataFrame, list[dict]]:
    corpus_path = Path(CORPUS_PATH)
    queries_path = Path(QUERIES_PATH)

    if corpus_path.exists() and queries_path.exists():
        print(f"[skip] {corpus_path} and {queries_path} already exist.")
        corpus_df = pd.read_parquet(corpus_path)
        with open(queries_path) as f:
            queries = json.load(f)
        print(f"  loaded {len(corpus_df)} passages, {len(queries)} queries")
        return corpus_df, queries

    dataset = ir_datasets.load("msmarco-passage/dev/small")

    print("Loading qrels...")
    qrels_by_query: dict[str, list[str]] = {}
    relevant_doc_ids: set[str] = set()
    for qrel in dataset.qrels_iter():
        if qrel.relevance <= 0:
            continue
        qrels_by_query.setdefault(qrel.query_id, []).append(qrel.doc_id)
        relevant_doc_ids.add(qrel.doc_id)
    print(f"  {len(relevant_doc_ids)} relevant passages across {len(qrels_by_query)} queries")

    docs_store = dataset.docs_store()

    print(f"Fetching text for {len(relevant_doc_ids)} relevant passages...")
    corpus: dict[str, str] = {}
    for doc_id in tqdm(sorted(relevant_doc_ids)):
        doc = docs_store.get(doc_id)
        if doc is not None:
            corpus[doc_id] = doc.text

    corpus = _fill_with_distractors(corpus, docs_store, CORPUS_SIZE)

    corpus_df = _save_corpus(corpus)
    queries = _save_queries_binary(corpus, qrels_by_query, dataset)
    return corpus_df, queries


# ---------------------------------------------------------------------------
# TREC Deep Learning 2019 / 2020 (graded relevance 0-3)
# ---------------------------------------------------------------------------

def build_corpus_and_queries_trec_dl(year: int) -> tuple[pd.DataFrame, list[dict]]:
    corpus_path = Path(CORPUS_PATH)
    queries_path = Path(QUERIES_PATH)

    if corpus_path.exists() and queries_path.exists():
        print(f"[skip] {corpus_path} and {queries_path} already exist.")
        corpus_df = pd.read_parquet(corpus_path)
        with open(queries_path) as f:
            queries = json.load(f)
        print(f"  loaded {len(corpus_df)} passages, {len(queries)} queries")
        return corpus_df, queries

    dataset_id = f"msmarco-passage/trec-dl-{year}/judged"
    print(f"Loading {dataset_id}...")
    dataset = ir_datasets.load(dataset_id)

    qrels_by_query: dict[str, dict[str, int]] = {}
    relevant_doc_ids: set[str] = set()
    all_judged_ids: set[str] = set()
    for qrel in dataset.qrels_iter():
        qrels_by_query.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance
        all_judged_ids.add(qrel.doc_id)
        if qrel.relevance >= 1:
            relevant_doc_ids.add(qrel.doc_id)

    hard_neg_ids = all_judged_ids - relevant_doc_ids
    print(
        f"  {len(qrels_by_query)} queries | "
        f"{len(relevant_doc_ids)} relevant passages | "
        f"{len(hard_neg_ids)} hard negatives (grade=0)"
    )

    # Passages live in the parent MS MARCO collection, not the TREC-DL subset
    docs_store = ir_datasets.load("msmarco-passage").docs_store()

    corpus: dict[str, str] = {}

    print(f"Fetching {len(relevant_doc_ids)} relevant passages (grade>=1)...")
    for doc_id in tqdm(sorted(relevant_doc_ids)):
        doc = docs_store.get(doc_id)
        if doc is not None:
            corpus[doc_id] = doc.text

    remaining = CORPUS_SIZE - len(corpus)
    print(f"Fetching up to {remaining} hard negatives (grade=0)...")
    for doc_id in tqdm(sorted(hard_neg_ids)):
        if len(corpus) >= CORPUS_SIZE:
            break
        if doc_id in corpus:
            continue
        doc = docs_store.get(doc_id)
        if doc is not None:
            corpus[doc_id] = doc.text

    corpus = _fill_with_distractors(corpus, docs_store, CORPUS_SIZE)

    corpus_df = _save_corpus(corpus)
    queries = _save_queries_graded(corpus, qrels_by_query, dataset)
    return corpus_df, queries


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fill_with_distractors(
    corpus: dict[str, str],
    docs_store,
    target_size: int,
) -> dict[str, str]:
    needed = target_size - len(corpus)
    if needed <= 0:
        if len(corpus) > target_size:
            print(f"Capping to {target_size} (had {len(corpus)})")
            corpus = dict(list(corpus.items())[:target_size])
        return corpus

    try:
        total_docs = ir_datasets.load("msmarco-passage").docs_count()
    except Exception:
        total_docs = 8_841_823

    print(f"Sampling {needed} random distractors from {total_docs} total...")
    rng = random.Random(RANDOM_SEED)
    pool_size = min(max(needed * 3, needed + 1000), total_docs)
    pool = rng.sample(range(total_docs), pool_size)
    with tqdm(total=needed) as pbar:
        for idx in pool:
            if len(corpus) >= target_size:
                break
            doc_id = str(idx)
            if doc_id in corpus:
                continue
            doc = docs_store.get(doc_id)
            if doc is not None:
                corpus[doc_id] = doc.text
                pbar.update(1)
    return corpus


def _save_corpus(corpus: dict[str, str]) -> pd.DataFrame:
    corpus_df = pd.DataFrame(
        [{"doc_id": did, "text": txt} for did, txt in corpus.items()]
    )
    corpus_path = Path(CORPUS_PATH)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_df.to_parquet(corpus_path, index=False)
    print(f"Wrote {corpus_path} ({len(corpus_df)} passages)")
    return corpus_df


def _save_queries_binary(
    corpus: dict[str, str],
    qrels_by_query: dict[str, list[str]],
    dataset,
) -> list[dict]:
    corpus_ids = set(corpus.keys())
    query_text_by_id = {q.query_id: q.text for q in dataset.queries_iter()}
    queries: list[dict] = []
    for qid, doc_ids in qrels_by_query.items():
        kept = [d for d in doc_ids if d in corpus_ids]
        if not kept or qid not in query_text_by_id:
            continue
        queries.append({
            "query_id": qid,
            "query_text": query_text_by_id[qid],
            "relevant_doc_ids": kept,
        })
    queries_path = Path(QUERIES_PATH)
    with open(queries_path, "w") as f:
        json.dump(queries, f, indent=2)
    print(f"Wrote {queries_path} ({len(queries)} queries)")
    return queries


def _save_queries_graded(
    corpus: dict[str, str],
    qrels_by_query: dict[str, dict[str, int]],
    dataset,
) -> list[dict]:
    corpus_ids = set(corpus.keys())
    query_text_by_id = {q.query_id: q.text for q in dataset.queries_iter()}
    queries: list[dict] = []
    for qid, grades in qrels_by_query.items():
        if qid not in query_text_by_id:
            continue
        # All judged passages that landed in our corpus (including grade=0)
        judged_in_corpus = {pid: g for pid, g in grades.items() if pid in corpus_ids}
        relevant_in_corpus = [pid for pid, g in judged_in_corpus.items() if g >= 1]
        if not relevant_in_corpus:
            continue
        queries.append({
            "query_id": qid,
            "query_text": query_text_by_id[qid],
            "relevant_doc_ids": relevant_in_corpus,
            "relevance_grades": judged_in_corpus,  # grade=0 entries penalise bad ranking
        })
    queries_path = Path(QUERIES_PATH)
    with open(queries_path, "w") as f:
        json.dump(queries, f, indent=2)
    print(f"Wrote {queries_path} ({len(queries)} queries, graded 0-3)")
    return queries


# ---------------------------------------------------------------------------
# ChromaDB indexing (shared)
# ---------------------------------------------------------------------------

def build_chroma_index(corpus_df: pd.DataFrame):
    import chromadb
    from sentence_transformers import SentenceTransformer

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    existing = {c.name for c in client.list_collections()}

    if COLLECTION_NAME in existing:
        coll = client.get_collection(COLLECTION_NAME)
        if coll.count() == len(corpus_df):
            print(f"[skip] chroma '{COLLECTION_NAME}' already has {coll.count()} docs.")
            return coll
        print(f"Recreating collection (had {coll.count()}, expected {len(corpus_df)})")
        client.delete_collection(COLLECTION_NAME)

    coll = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    doc_ids = corpus_df["doc_id"].tolist()
    texts = corpus_df["text"].tolist()
    n_batches = (len(texts) + EMBED_BATCH - 1) // EMBED_BATCH
    print(f"Embedding + indexing {len(texts)} passages in {n_batches} batches...")
    for bi in tqdm(range(n_batches)):
        lo = bi * EMBED_BATCH
        hi = lo + EMBED_BATCH
        embs = model.encode(texts[lo:hi], show_progress_bar=False, convert_to_numpy=True)
        coll.add(ids=doc_ids[lo:hi], documents=texts[lo:hi], embeddings=embs.tolist())

    return coll


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["msmarco-dev-small", "trec-dl-2019", "trec-dl-2020"],
        default="msmarco-dev-small",
        help="Which dataset to build the corpus from (default: msmarco-dev-small)",
    )
    args = parser.parse_args()

    if args.dataset == "trec-dl-2019":
        corpus_df, queries = build_corpus_and_queries_trec_dl(2019)
    elif args.dataset == "trec-dl-2020":
        corpus_df, queries = build_corpus_and_queries_trec_dl(2020)
    else:
        corpus_df, queries = build_corpus_and_queries_msmarco()

    coll = build_chroma_index(corpus_df)

    print()
    print("=== Summary ===")
    print(f"Dataset:                  {args.dataset}")
    print(f"Corpus size:              {len(corpus_df)}")
    print(f"Queries with qrels:       {len(queries)}")
    print(f"Chroma collection count:  {coll.count()}")


if __name__ == "__main__":
    main()
