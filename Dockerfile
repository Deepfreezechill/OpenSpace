# ============================================
# OpenSpace Multi-Stage Dockerfile
# ============================================
# Build: docker build -t openspace .
# Run:   docker run -p 8000:8000 --env-file .env openspace
#
# Environment variables (see .env.example):
#   OPENSPACE_MCP_HOST, OPENSPACE_MCP_PORT, OPENSPACE_MCP_TRANSPORT,
#   OPENSPACE_LOG_LEVEL, OPENSPACE_SHUTDOWN_TIMEOUT,
#   OPENSPACE_METRICS_ENABLED, OPENROUTER_API_KEY, etc.
# ============================================

# ── Stage 1: Build ──────────────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency files first (layer caching)
COPY pyproject.toml requirements.txt ./
COPY openspace/ openspace/

# Install Python dependencies
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt && \
    pip install --no-cache-dir --prefix=/install .

# ── Stage 2: Runtime ────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

LABEL org.opencontainers.image.title="OpenSpace" \
      org.opencontainers.image.description="AI Agent Framework with MCP Server" \
      org.opencontainers.image.source="https://github.com/Deepfreezechill/OpenSpace"

# Create non-root user
RUN groupadd --gid 1001 openspace && \
    useradd --uid 1001 --gid openspace --shell /bin/bash --create-home openspace

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
WORKDIR /app
COPY openspace/ openspace/
COPY pyproject.toml ./

# Create directories for runtime data
RUN mkdir -p /app/skills /app/logs && \
    chown -R openspace:openspace /app

# Switch to non-root user
USER openspace

# Default environment
ENV OPENSPACE_MCP_HOST=0.0.0.0 \
    OPENSPACE_MCP_PORT=8000 \
    OPENSPACE_MCP_TRANSPORT=streamable-http \
    OPENSPACE_LOG_LEVEL=INFO \
    OPENSPACE_SHUTDOWN_TIMEOUT=30 \
    OPENSPACE_METRICS_ENABLED=true \
    OPENSPACE_SKILL_STORE_PATH=/app/skills \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Health check: hit the MCP health endpoint
# For stdio transport, override this in docker-compose
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Graceful shutdown: Docker sends SIGTERM, handler drains in-flight tasks
STOPSIGNAL SIGTERM

ENTRYPOINT ["python", "-m", "openspace.mcp_server"]
CMD ["--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8000"]
