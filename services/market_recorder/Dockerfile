# ============================================================================
# Market Recorder Service - Production Dockerfile
# ============================================================================
# Multi-stage build for smaller image and better caching

# Stage 1: Build dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install to /install
COPY services/market_recorder/requirements.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# Stage 2: Production image
FROM python:3.11-slim

# Labels for container registry
LABEL org.opencontainers.image.title="Market Recorder"
LABEL org.opencontainers.image.description="Real-time crypto market data streaming with S/R penetration signals"
LABEL org.opencontainers.image.version="0.2.0"

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY services/ ./services/
COPY src/ ./src/

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with production settings
# - 4 workers for concurrency
# - 120s timeout for long-running requests
# - Graceful shutdown
CMD ["uvicorn", "services.market_recorder.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "120", \
     "--access-log", \
     "--proxy-headers"]
