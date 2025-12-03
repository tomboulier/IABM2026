.PHONY: install run test docker-build docker-run clean

# Environment setup using uv
install:
	uv venv
	uv pip install -r pyproject.toml

# Run the main experiment
run:
	uv run python main.py

# Run tests (if any)
test:
	uv run python -m unittest discover -s . -p "test_*.py"

# Docker build
docker-build:
	docker build -t medmnist-variability .

# Docker run
docker-run:
	docker run --rm medmnist-variability

# Clean up
clean:
	rm -rf .venv
	rm -rf __pycache__
	rm -rf .pytest_cache
