VENV ?= .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: install load-data load-data-trec-dl reset-data reset-data-trec-dl build-data serve benchmark benchmark-corpus benchmark-trec-dl test clean

install:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

load-data:
	$(PY) scripts/load_corpus.py

load-data-trec-dl:
	$(PY) scripts/load_corpus.py --dataset trec-dl-2019

reset-data:
	rm -rf data
	mkdir -p data
	$(PY) scripts/load_corpus.py

reset-data-trec-dl:
	rm -rf data
	mkdir -p data
	$(PY) scripts/load_corpus.py --dataset trec-dl-2019

build-data:
	$(PY) scripts/build_gold_set.py

serve:
	$(VENV)/bin/uvicorn src.api:app --reload --port 8000

benchmark:
	$(PY) scripts/benchmark.py

benchmark-corpus:
	$(PY) scripts/benchmark_corpus.py

benchmark-trec-dl: reset-data-trec-dl
	$(PY) scripts/benchmark_corpus.py

test:
	$(VENV)/bin/pytest tests/ -v

clean:
	rm -rf data/chroma_db results/*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
