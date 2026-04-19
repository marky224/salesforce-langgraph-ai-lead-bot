# ----------------------------------------------------------------------------
# Stage 1: Build dependencies
# ----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ----------------------------------------------------------------------------
# Stage 2: Production image
# ----------------------------------------------------------------------------
FROM python:3.12-slim

LABEL maintainer="Mark Marquez"
LABEL project="salesforce-langgraph-ai-lead-bot"

# Run as non-root
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY app/ ./app/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Uvicorn with 1 worker (free tier), 120s timeout for LLM streaming
CMD ["python", "-m", "uvicorn", "app.server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "120", \
     "--log-level", "info"]
