.PHONY: install test lint check-robots sample-run clean

install:
	uv pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=ingestion_source

check-robots:
	ingest check-robots --url https://indiankanoon.org/doc/1940428/

sample-run:
	@echo "Running polite sample acquisition for Delhi High Court (5 judgments)..."
	ingest run --court "Delhi High Court" --year 2023 --mode hybrid --limit 5

clean:
	@rm -rf raw-store/ .pytest_cache/ .coverage .pytest_cache/ __pycache__/ src/ingestion_source/__pycache__/ tests/__pycache__/
