# Python 3.13 Linux environment
FROM python:3.13-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

# Copy dependency configuration
COPY pyproject.toml uv.lock ./

# Install locked dependencies
RUN uv sync --locked --no-dev --no-install-project

# Install CUDA-compatible torch_scatter
RUN uv pip install --python /app/.venv/bin/python \
    torch-scatter \
    --no-index \
    --find-links https://data.pyg.org/whl/torch-2.11.0+cu128.html \
    --only-binary=:all:

# Copy application code, checkpoint, and stats
COPY . .

# Use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Expose FastAPI
EXPOSE 8000

# Start inference API
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]