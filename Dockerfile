FROM python:3.11-slim

WORKDIR /app

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml README.md uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy source code and config
COPY src/ ./src/
COPY config/ ./config/

# Create models directory
RUN mkdir -p models

# Expose port
EXPOSE 8000

# Start FastAPI
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
