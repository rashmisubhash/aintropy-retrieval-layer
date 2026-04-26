"""Build TWO paraphrase test sets from real MS MARCO qrel clusters.

Design intent
-------------
A qrel cluster is a single MS MARCO passage that multiple human-authored
queries are all judged relevant to. That makes those queries real paraphrases
of each other, by construction. We combine train + dev/small qrels and then
produce two files with two different purposes:

  data/paraphrases_cache_test.json
      Purpose: measure semantic-cache HIT RATE.
      Source:  all qrel clusters of size >= 2 (~13,515 unfiltered).
      We deterministically sample 500 clusters. The shared doc does NOT need
      to be in our indexed corpus — we are only asking "does the embedding
      model place paraphrases close enough to clear the threshold?" so the
      retrieval pipeline's correctness is irrelevant here.

  data/paraphrases_recall_test.json
      Purpose: measure RECALL PRESERVATION after a semantic-cache hit.
      Source:  qrel clusters of size >= 2 filtered to docs that are in our
               indexed 10k corpus (~313 total). Coverage is tight so we use
               every cluster (no sampling).
      Each entry carries shared_relevant_doc_ids so the benchmark can check
      whether the cached top-5 (served to a paraphrase) still contains a
      doc that is relevant to the paraphrase.

Why the split: entangling hit-rate and recall measurements in a single file
conflates "did the cache fire?" with "did the cached answer happen to be
correct?". Separating them gives two clean numbers instead of one muddy one.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import ir_datasets
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CORPUS_PATH  # noqa: E402

RANDOM_SEED = 42
N_CACHE_SAMPLES = 500
MIN_CLUSTER_SIZE = 2

CACHE_TEST_PATH = "data/paraphrases_cache_test.json"
RECALL_TEST_PATH = "data/paraphrases_recall_test.json"

SPLITS = ["msmarco-passage/train", "msmarco-passage/dev/small"]


def load_qrels_and_queries() -> tuple[dict[str, set[str]], dict[str, str]]:
    doc_to_queries: dict[str, set[str]] = defaultdict(set)
    query_text_by_id: dict[str, str] = {}
    for split in SPLITS:
        ds = ir_datasets.load(split)
        print(f"Loading qrels from {split}...")
        for q in ds.qrels_iter():
            if q.relevance > 0:
                doc_to_queries[q.doc_id].add(q.query_id)
        print(f"Loading queries from {split}...")
        for q in ds.queries_iter():
            query_text_by_id[q.query_id] = q.text
    return doc_to_queries, query_text_by_id


def build_records(
    doc_ids: list[str],
    doc_to_queries: dict[str, set[str]],
    query_text_by_id: dict[str, str],
    include_shared_doc: bool,
) -> list[dict]:
    records: list[dict] = []
    for doc_id in doc_ids:
        qids = sorted(doc_to_queries[doc_id])
        # Skip clusters where any query has no text mapping (should be rare).
        if not all(qid in query_text_by_id for qid in qids):
            continue
        original_qid, *paraphrase_qids = qids
        rec: dict = {
            "original_query_id": original_qid,
            "original_text": query_text_by_id[original_qid],
            "paraphrase_query_ids": paraphrase_qids,
            "paraphrase_texts": [query_text_by_id[q] for q in paraphrase_qids],
        }
        if include_shared_doc:
            rec["shared_relevant_doc_ids"] = [doc_id]
        records.append(rec)
    return records


def main() -> None:
    doc_to_queries, query_text_by_id = load_qrels_and_queries()
    print(f"  total qrelled docs:   {len(doc_to_queries):>7,}")
    print(f"  total queries w/text: {len(query_text_by_id):>7,}")

    corpus_ids = set(pd.read_parquet(CORPUS_PATH)["doc_id"].astype(str))
    print(f"  indexed corpus size:  {len(corpus_ids):>7,}")

    all_multi = sorted(
        d for d, qs in doc_to_queries.items() if len(qs) >= MIN_CLUSTER_SIZE
    )
    corpus_multi = sorted(d for d in all_multi if d in corpus_ids)
    print(f"  unfiltered clusters (size>={MIN_CLUSTER_SIZE}): {len(all_multi):>7,}")
    print(f"  corpus-filtered clusters:              {len(corpus_multi):>7,}")

    rng = random.Random(RANDOM_SEED)
    cache_sample = rng.sample(all_multi, min(N_CACHE_SAMPLES, len(all_multi)))
    cache_records = build_records(
        cache_sample, doc_to_queries, query_text_by_id, include_shared_doc=False
    )

    recall_records = build_records(
        corpus_multi, doc_to_queries, query_text_by_id, include_shared_doc=True
    )

    Path(CACHE_TEST_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_TEST_PATH, "w") as f:
        json.dump(cache_records, f, indent=2)
    with open(RECALL_TEST_PATH, "w") as f:
        json.dump(recall_records, f, indent=2)

    def total_pairs(recs: list[dict]) -> int:
        return sum(len(r["paraphrase_query_ids"]) for r in recs)

    print()
    print("=== Summary ===")
    print(
        f"{CACHE_TEST_PATH}:  {len(cache_records):>4} clusters, "
        f"{total_pairs(cache_records):>5} paraphrase pairs"
    )
    print(
        f"{RECALL_TEST_PATH}: {len(recall_records):>4} clusters, "
        f"{total_pairs(recall_records):>5} paraphrase pairs"
    )

    def show(label: str, records: list[dict], n: int = 3) -> None:
        print()
        print(f"=== {n} examples from {label} ===")
        for rec in records[:n]:
            extras = (
                f"  shared_doc={rec['shared_relevant_doc_ids']}"
                if "shared_relevant_doc_ids" in rec
                else ""
            )
            print(f"  qid={rec['original_query_id']}{extras}")
            print(f"    original        : {rec['original_text']}")
            for qid, t in zip(rec["paraphrase_query_ids"], rec["paraphrase_texts"]):
                print(f"    para (q={qid:>7}) : {t}")
            print()

    show(CACHE_TEST_PATH, cache_records)
    show(RECALL_TEST_PATH, recall_records)


if __name__ == "__main__":
    main()
