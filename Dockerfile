FROM python:3.11-slim

WORKDIR /app

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy everything
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
COPY config/ ./config/
COPY mlruns/ ./mlruns/
COPY models/ ./models/

# Install dependencies
RUN uv sync --frozen --no-dev

# Expose port
EXPOSE 8000

# Start FastAPI
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
