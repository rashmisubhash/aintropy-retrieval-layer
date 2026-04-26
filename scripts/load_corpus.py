"""Download MS MARCO dev/small, build a 10k-passage corpus, and index with ChromaDB.

Idempotent:
  - Skips corpus/queries build if data/corpus.parquet and data/queries.json both exist.
  - Skips Chroma indexing if the collection already has the expected doc count.
"""

from __future__ import annotations  # for Python 3.10+ type hinting (e.g. dict[str, list[str]])

import json
import random
import sys
from pathlib import Path

import ir_datasets
import pandas as pd
from tqdm import tqdm

# Allow `from src.config import ...` when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    CHROMA_PATH,
    CORPUS_PATH,
    CORPUS_SIZE,
    EMBEDDING_MODEL,
    QUERIES_PATH,
)

RANDOM_SEED = 42    # for reproducible sampling of distractor passages
EMBED_BATCH = 256
COLLECTION_NAME = "msmarco_passages"


def build_corpus_and_queries() -> tuple[pd.DataFrame, list[dict]]:
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
    print(
        f"  {len(relevant_doc_ids)} relevant passages across {len(qrels_by_query)} queries"
    )

    docs_store = dataset.docs_store()

    print(f"Fetching text for {len(relevant_doc_ids)} relevant passages...")
    corpus: dict[str, str] = {}
    for doc_id in tqdm(sorted(relevant_doc_ids)):
        doc = docs_store.get(doc_id)
        if doc is not None:
            corpus[doc_id] = doc.text

    needed = CORPUS_SIZE - len(corpus)
    if needed > 0:
        try:
            total_docs = dataset.docs_count()
        except Exception:
            total_docs = 8_841_823
        print(f"Sampling {needed} distractor passages from {total_docs} total...")
        rng = random.Random(RANDOM_SEED)
        pool_size = min(max(needed * 3, needed + 1000), total_docs)
        pool = rng.sample(range(total_docs), pool_size)
        with tqdm(total=needed) as pbar:
            for idx in pool:
                if len(corpus) >= CORPUS_SIZE:
                    break
                doc_id = str(idx)
                if doc_id in corpus:
                    continue
                doc = docs_store.get(doc_id)
                if doc is not None:
                    corpus[doc_id] = doc.text
                    pbar.update(1)
    elif len(corpus) > CORPUS_SIZE:
        print(f"Capping to {CORPUS_SIZE} (had {len(corpus)} relevant passages)")
        corpus = dict(list(corpus.items())[:CORPUS_SIZE])

    print(f"Final corpus size: {len(corpus)}")
    corpus_df = pd.DataFrame(
        [{"doc_id": did, "text": txt} for did, txt in corpus.items()]
    )
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_df.to_parquet(corpus_path, index=False)
    print(f"Wrote {corpus_path}")

    corpus_ids = set(corpus.keys())
    print("Loading query texts...")
    query_text_by_id = {q.query_id: q.text for q in dataset.queries_iter()}

    queries: list[dict] = []
    for qid, doc_ids in qrels_by_query.items():
        kept = [d for d in doc_ids if d in corpus_ids]
        if not kept or qid not in query_text_by_id:
            continue
        queries.append(
            {
                "query_id": qid,
                "query_text": query_text_by_id[qid],
                "relevant_doc_ids": kept,
            }
        )

    with open(queries_path, "w") as f:
        json.dump(queries, f, indent=2)
    print(f"Wrote {queries_path} ({len(queries)} queries)")
    return corpus_df, queries


def build_chroma_index(corpus_df: pd.DataFrame):
    import chromadb
    from sentence_transformers import SentenceTransformer

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    existing = {c.name for c in client.list_collections()}

    if COLLECTION_NAME in existing:
        coll = client.get_collection(COLLECTION_NAME)
        if coll.count() == len(corpus_df):
            print(
                f"[skip] chroma collection '{COLLECTION_NAME}' already has "
                f"{coll.count()} docs."
            )
            return coll
        print(
            f"Recreating collection (had {coll.count()}, expected {len(corpus_df)})"
        )
        client.delete_collection(COLLECTION_NAME)

    coll = client.create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    doc_ids = corpus_df["doc_id"].tolist()
    texts = corpus_df["text"].tolist()

    n_batches = (len(texts) + EMBED_BATCH - 1) // EMBED_BATCH
    print(
        f"Embedding + indexing {len(texts)} passages "
        f"in {n_batches} batches of {EMBED_BATCH}..."
    )
    for bi in tqdm(range(n_batches)):
        lo = bi * EMBED_BATCH
        hi = lo + EMBED_BATCH
        batch_ids = doc_ids[lo:hi]
        batch_texts = texts[lo:hi]
        embs = model.encode(
            batch_texts, show_progress_bar=False, convert_to_numpy=True
        )
        coll.add(ids=batch_ids, documents=batch_texts, embeddings=embs.tolist())

    return coll


def main() -> None:
    corpus_df, queries = build_corpus_and_queries()
    coll = build_chroma_index(corpus_df)

    print()
    print("=== Summary ===")
    print(f"Corpus size:              {len(corpus_df)}")
    print(f"Queries with qrels:       {len(queries)}")
    print(f"Chroma collection count:  {coll.count()}")


if __name__ == "__main__":
    main()
