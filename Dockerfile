# GraphFusion backend (FastAPI + DuckDB)
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DFG_WORKSPACE=/app/workspace \
    DFG_LOG_JSON=true

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
RUN pip install -r requirements.txt

COPY backend ./backend
COPY config ./config
COPY scripts ./scripts
COPY experiments ./experiments
COPY tests ./tests
COPY data/sample ./data/sample
RUN pip install --no-deps -e . && mkdir -p /app/workspace /app/data/raw /app/data/processed

RUN useradd --create-home --uid 10001 dfg && chown -R dfg:dfg /app
USER dfg

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
