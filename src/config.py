"""Project-wide constants: model names, paths, retrieval thresholds."""

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CORPUS_SIZE = 10000
VECTOR_TOP_K = 50
FINAL_TOP_K = 5
LATENCY_BUDGET_MS = 500
RERANK_SKIP_THRESHOLD_MS = 400

SEMANTIC_CACHE_THRESHOLD = 0.92
CACHE_MAX_SIZE = 1000
CACHE_TTL_HOURS = 24

CORPUS_VERSION = "msmarco-1k-v1"

CHROMA_PATH = "data/chroma_db"
CORPUS_PATH = "data/corpus.parquet"
QUERIES_PATH = "data/queries.json"
