FROM python:3.13-slim

WORKDIR /app

# Install runtime and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install UV package manager
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using UV
RUN uv sync --group prod

# Copy application code
COPY scripts/ ./scripts/
COPY tests_e2e/ ./tests_e2e/

# Create directories for data and results
RUN mkdir -p data/datasets data/experiments results

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Make scripts executable
RUN chmod +x scripts/*.py tests_e2e/*.py

# Default to Python shell for flexible script execution
CMD ["python3"]