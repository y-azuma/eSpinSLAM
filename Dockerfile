FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    NUMBA_CACHE_DIR=/tmp/numba-cache

RUN apt-get update \
    && apt-get install --yes --no-install-recommends git libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/repo
COPY pyproject.toml README.md ./
COPY espinslam ./espinslam
COPY scripts ./scripts
COPY examples ./examples
RUN pip install --no-cache-dir ".[topspin]"

CMD ["python", "scripts/evaluate_topspin.py", "--help"]
