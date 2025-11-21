FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY datasets.py .
COPY metrics.py .
COPY models.py .
COPY main.py .

# Install dependencies
RUN uv venv && uv pip install -r pyproject.toml

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"

# Entrypoint
CMD ["python", "main.py"]
