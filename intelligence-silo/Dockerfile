FROM python:3.11-slim

WORKDIR /app

# System dependencies for PyTorch + FAISS compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (avoids pulling the massive CUDA bundle ~2.5GB → ~750MB)
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install package dependencies (production only — no dev extras)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e "." \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy source
COPY . .

# Create data directory for persistent storage (mount a volume here in Cloud Run)
RUN mkdir -p /data/semantic_index /data/decision_journal /data/checkpoints

# Non-root user for security
RUN useradd --system --no-create-home --uid 1001 silo \
    && chown -R silo:silo /app /data
USER silo

EXPOSE 8080

# Cloud Run sends SIGTERM; uvicorn handles graceful shutdown by default
CMD ["uvicorn", "core.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--timeout-graceful-shutdown", "30", \
     "--log-level", "info"]
