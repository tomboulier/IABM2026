.PHONY: install run test docker-build docker-run clean

# Environment setup using uv
install:
	uv venv
	uv sync

# Run the main experiment
run:
	uv run python main.py

# Run tests (if any)
test:
	uv run pytest

# Docker build
docker-build:
	docker build -t medmnist-variability-similarity .

# Docker run
docker-run:
	docker run --rm medmnist-variability-similarity

# Clean up
clean:
	rm -rf .venv
	rm -rf __pycache__
	rm -rf .pytest_cache
