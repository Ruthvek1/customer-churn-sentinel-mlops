# ============================================================
# Multi-stage Dockerfile for Churn Prediction ML System
# ============================================================
# Stage 1: Builder — install dependencies
# Stage 2: Production — lean runtime image
# ============================================================

# --- Stage 1: Builder ---
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# --- Stage 2: Production ---
FROM python:3.11-slim AS production

WORKDIR /app

# Create non-root user for security
RUN groupadd -r mluser && useradd -r -g mluser mluser

# Copy installed packages from builder
COPY --from=builder /root/.local /home/mluser/.local
ENV PATH=/home/mluser/.local/bin:$PATH

# Copy application code
COPY src/ ./src/
COPY dashboard/ ./dashboard/
COPY config/ ./config/
COPY data/ ./data/
COPY models/ ./models/
COPY logs/ ./logs/

# Create necessary directories
RUN mkdir -p data/raw data/processed data/reference models logs drift_reports && \
    chown -R mluser:mluser /app

# Switch to non-root user
USER mluser

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

# Default command: start FastAPI
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
