.PHONY: install check lint types test fmt clean

BACKEND := backend

# The same set .github/workflows/ci.yml formats and lints, so one list keeps a
# green `make check` and a green CI from drifting apart.
PY_FILES := src tests analyse_failures.py annotate.py build_eval_subset.py build_public_results.py categorise.py llm_smoke.py run_evaluation.py seed_demo.py sweep_threshold.py transactions.py users.py validate_dataset.py

install:
	cd $(BACKEND) && uv sync

fmt:
	cd $(BACKEND) && uv run ruff format $(PY_FILES) && uv run ruff check --fix $(PY_FILES)

lint:
	cd $(BACKEND) && uv run ruff format --check $(PY_FILES) && uv run ruff check $(PY_FILES)

types:
	cd $(BACKEND) && uv run mypy

test:
	cd $(BACKEND) && uv run pytest

arch:
	cd $(BACKEND) && uv run lint-imports

# The milestone-0 definition of done: one command runs lint, types, and tests.
check: lint types arch test

clean:
	cd $(BACKEND) && rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
