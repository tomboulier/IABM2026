.PHONY: install run test docker-build docker-run clean abstract abstract-clean

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

# Compile abstract PDF
abstract:
	cd abstract && pdflatex abstract.tex && bibtex abstract && pdflatex abstract.tex && pdflatex abstract.tex
	@echo "PDF generated: abstract/abstract.pdf"

# Clean abstract build files
abstract-clean:
	cd abstract && rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz
