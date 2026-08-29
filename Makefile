.DEFAULT_GOAL := help
.PHONY: help install install-dev generate-data run app test test-eval test-fast lint format typecheck coverage clean all

PY ?= python
SEED ?= 42
N_RECORDS ?= 60

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Install the core package (offline stack, no ML wheels)
	$(PY) -m pip install -e .

install-dev:  ## Install everything: embeddings, llm, app, dev tooling
	$(PY) -m pip install -e ".[all]"

generate-data:  ## Regenerate the synthetic batch + ground truth
	$(PY) scripts/generate_data.py --n-records $(N_RECORDS) --seed $(SEED)

run: generate-data  ## Generate data, then reconcile it
	$(PY) scripts/run_reconciliation.py

app:  ## Launch the Streamlit dashboard
	$(PY) -m streamlit run app.py

test:  ## Full test suite
	$(PY) -m pytest

test-fast:  ## Skip anything touching the real embedding/LLM stack
	$(PY) -m pytest -m "not slow" -q

test-eval:  ## Print the match-rate / accuracy report
	$(PY) -m pytest tests/evaluation -s

coverage:  ## Test suite with a coverage report
	$(PY) -m pytest --cov --cov-report=term-missing --cov-report=html

lint:  ## Check style and common bugs
	$(PY) -m ruff check .

format:  ## Auto-fix what can be auto-fixed
	$(PY) -m ruff check . --fix
	$(PY) -m ruff format .

clean:  ## Remove caches and generated artifacts
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	rm -rf data/outputs/* data/raw/*.csv data/processed/*.json
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true

all: lint test  ## Lint and test — what CI runs
