"""Build a 1k-passage MS MARCO corpus + a 200-query gold set with graded relevance.

NOTE on MS MARCO sparsity
-------------------------
MS MARCO qrels are sparse: by annotation design each query has roughly one
relevant passage (avg ~1.07 in dev/small). Recall@k computed against this gold
set will therefore look artificially low because the gold set undercounts the
true number of relevant passages present in the corpus. This is a known
annotation artifact, not a retrieval failure. Document this caveat in the
README before reporting recall metrics.

Outputs
-------
data/gold_set_corpus.jsonl — exactly 1,000 passages, length-stratified into 10
                        buckets of 100 by (len(passage) // 100) % 10.
data/gold_set.jsonl   — 200 queries (default), each with >=1 relevant passage
                        in the corpus. Top-similarity passage per query gets
                        grade=2, rest get grade=1, so each query has exactly
                        one grade=2 entry.

CLI
---
    python scripts/build_gold_set.py [--seed N]

Re-run with a different --seed if fewer than 150 eligible queries are found.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import ir_datasets
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import EMBEDDING_MODEL  # noqa: E402

CORPUS_PATH = Path("data/gold_set_corpus.jsonl")
GOLD_PATH = Path("data/gold_set.jsonl")

N_BUCKETS = 10
PASSAGES_PER_BUCKET = 100
TARGET_CORPUS_SIZE = N_BUCKETS * PASSAGES_PER_BUCKET  # 1000
TARGET_QUERIES = 200
MIN_QUERIES = 150


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build 1k corpus + 200-query gold set.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print("Loading MS MARCO dev/small qrels...")
    ds = ir_datasets.load("msmarco-passage/dev/small")
    docs_store = ds.docs_store()

    query_to_relevant: dict[str, set[str]] = defaultdict(set)
    relevant_passage_ids: set[str] = set()
    for q in ds.qrels_iter():
        if q.relevance == 1:
            query_to_relevant[q.query_id].add(q.doc_id)
            relevant_passage_ids.add(q.doc_id)
    print(f"  unique relevant passages: {len(relevant_passage_ids):,}")
    print(f"  queries with >=1 qrel:    {len(query_to_relevant):,}")

    # ---- Step 1: 1k corpus, stratified by length bucket ----
    print("\nFetching passage text for stratification...")
    passages: dict[str, str] = {}
    for pid in sorted(relevant_passage_ids):
        doc = docs_store.get(pid)
        if doc is not None:
            passages[pid] = doc.text
    print(f"  fetched {len(passages):,} passage texts")

    buckets: dict[int, list[str]] = defaultdict(list)
    for pid, text in passages.items():
        bucket = (len(text) // 100) % 10
        buckets[bucket].append(pid)

    print("\nLength-bucket distribution (relevant passages):")
    for b in range(N_BUCKETS):
        print(f"  bucket {b}: {len(buckets[b]):,}")

    sampled: list[str] = []
    deficit = 0
    for b in range(N_BUCKETS):
        avail = sorted(buckets[b])
        take = min(PASSAGES_PER_BUCKET, len(avail))
        if take > 0:
            idx = rng.permutation(len(avail))[:take]
            sampled.extend(avail[int(i)] for i in idx)
        deficit += PASSAGES_PER_BUCKET - take

    if deficit > 0:
        already = set(sampled)
        leftovers = sorted(p for p in passages if p not in already)
        n_take = min(deficit, len(leftovers))
        idx = rng.permutation(len(leftovers))[:n_take]
        sampled.extend(leftovers[int(i)] for i in idx)
        print(f"\n  Bucket deficit of {deficit}; topped up {n_take} from leftovers.")

    sampled = sorted(set(sampled))
    if len(sampled) > TARGET_CORPUS_SIZE:
        sampled = sampled[:TARGET_CORPUS_SIZE]

    corpus_records = [
        {"passage_id": pid, "passage": passages[pid]} for pid in sampled
    ]
    print(f"\nCorpus size: {len(corpus_records):,}")

    # ---- Step 2: gold set from queries pointing at corpus passages ----
    corpus_pid_set = set(sampled)
    qid_to_relevant_in_corpus: dict[str, list[str]] = {}
    for qid, rel in query_to_relevant.items():
        in_corpus = sorted(rel & corpus_pid_set)
        if in_corpus:
            qid_to_relevant_in_corpus[qid] = in_corpus

    eligible_qids = sorted(qid_to_relevant_in_corpus.keys())
    print(f"  eligible queries (>=1 relevant passage in corpus): {len(eligible_qids):,}")

    if len(eligible_qids) < MIN_QUERIES:
        print(
            f"\n⚠️  Only {len(eligible_qids)} queries found. "
            f"Re-run with a different random seed (pass --seed N as CLI arg)."
        )
        sys.exit(1)

    sample_size = min(TARGET_QUERIES, len(eligible_qids))
    sampled_qid_indices = rng.permutation(len(eligible_qids))[:sample_size]
    sampled_qids = sorted(eligible_qids[int(i)] for i in sampled_qid_indices)
    print(f"  sampled queries:                                    {len(sampled_qids)}")

    print("\nLoading query texts...")
    query_text_by_id = {q.query_id: q.text for q in ds.queries_iter()}

    print(f"\nEmbedding queries + relevant passages with {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    query_texts = [query_text_by_id[qid] for qid in sampled_qids]
    query_embs = model.encode(query_texts, show_progress_bar=False, convert_to_numpy=True)
    query_embs = query_embs / np.linalg.norm(query_embs, axis=1, keepdims=True)

    all_rel_pids = sorted(
        {pid for qid in sampled_qids for pid in qid_to_relevant_in_corpus[qid]}
    )
    rel_texts = [passages[pid] for pid in all_rel_pids]
    rel_embs = model.encode(rel_texts, show_progress_bar=False, convert_to_numpy=True)
    rel_embs = rel_embs / np.linalg.norm(rel_embs, axis=1, keepdims=True)
    pid_to_idx = {pid: i for i, pid in enumerate(all_rel_pids)}

    gold_records: list[dict] = []
    for i, qid in enumerate(sampled_qids):
        rel_pids = qid_to_relevant_in_corpus[qid]
        idxs = [pid_to_idx[p] for p in rel_pids]
        sims = rel_embs[idxs] @ query_embs[i]
        order = np.argsort(-sims)
        ranked = [rel_pids[int(k)] for k in order]
        relevant_passages = []
        for rank, pid in enumerate(ranked):
            grade = 2 if rank == 0 else 1
            relevant_passages.append(
                {"passage_id": pid, "relevance": 1, "grade": grade}
            )
        gold_records.append(
            {
                "query_id": qid,
                "query": query_text_by_id[qid],
                "relevant_passages": relevant_passages,
            }
        )

    # ---- Step 3: validate before writing ----
    errors: list[str] = []
    if len(corpus_records) != TARGET_CORPUS_SIZE:
        errors.append(
            f"corpus size != {TARGET_CORPUS_SIZE} (got {len(corpus_records)})"
        )
    if len(gold_records) < MIN_QUERIES:
        errors.append(
            f"gold set query count < {MIN_QUERIES} (got {len(gold_records)})"
        )

    corpus_id_set = {r["passage_id"] for r in corpus_records}
    for g in gold_records:
        if not g["relevant_passages"]:
            errors.append(f"query {g['query_id']} has zero relevant passages")
            continue
        for rp in g["relevant_passages"]:
            if rp["passage_id"] not in corpus_id_set:
                errors.append(
                    f"passage {rp['passage_id']} in gold not in corpus "
                    f"(query {g['query_id']})"
                )
                break

    grade_counts: dict[int, int] = defaultdict(int)
    for g in gold_records:
        for rp in g["relevant_passages"]:
            grade_counts[rp["grade"]] += 1

    if grade_counts.get(2, 0) != len(gold_records):
        errors.append(
            f"grade=2 count {grade_counts.get(2, 0)} != query count "
            f"{len(gold_records)} (each query must have exactly one top passage)"
        )

    if errors:
        print("\nValidation failures:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(2)

    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CORPUS_PATH, "w") as f:
        for rec in corpus_records:
            f.write(json.dumps(rec) + "\n")
    with open(GOLD_PATH, "w") as f:
        for rec in gold_records:
            f.write(json.dumps(rec) + "\n")

    multi_rel = sum(1 for g in gold_records if len(g["relevant_passages"]) >= 2)
    referenced_pids = {
        rp["passage_id"] for g in gold_records for rp in g["relevant_passages"]
    }
    total_judgments = sum(len(g["relevant_passages"]) for g in gold_records)
    coverage_pct = len(referenced_pids) / len(corpus_records) * 100

    print("\n=== Summary ===")
    print(f"Corpus size:                          {len(corpus_records)}")
    print(f"Gold set queries:                     {len(gold_records)}")
    print(f"Total relevance judgments:            {total_judgments}")
    print(
        f"Queries with >=2 relevant passages:   {multi_rel}  "
        f"<- these make NDCG grading meaningful"
    )
    print(f"Grade distribution:                   {dict(grade_counts)}")
    print(
        f"Corpus coverage:                      "
        f"{len(referenced_pids)} passages referenced "
        f"({len(referenced_pids)}/{len(corpus_records)} = {coverage_pct:.1f}%)"
    )


if __name__ == "__main__":
    main()
