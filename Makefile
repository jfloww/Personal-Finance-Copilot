.PHONY: install check lint types test fmt clean

BACKEND := backend

install:
	cd $(BACKEND) && uv sync

fmt:
	cd $(BACKEND) && uv run ruff format src tests annotate.py build_eval_subset.py llm_smoke.py run_evaluation.py transactions.py validate_dataset.py && uv run ruff check --fix src tests annotate.py build_eval_subset.py llm_smoke.py run_evaluation.py transactions.py validate_dataset.py 

lint:
	cd $(BACKEND) && uv run ruff format --check src tests annotate.py build_eval_subset.py llm_smoke.py run_evaluation.py transactions.py validate_dataset.py && uv run ruff check src tests annotate.py build_eval_subset.py llm_smoke.py run_evaluation.py transactions.py validate_dataset.py 

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
