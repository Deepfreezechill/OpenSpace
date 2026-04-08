# ============================================
# Scion Multi-Stage Dockerfile
# ============================================
# Build: docker build -t scion .
# Run:   docker run -p 8000:8000 --env-file .env scion
#
# Environment variables (see scion/.env.example):
#   SCION_MCP_BEARER_TOKEN (REQUIRED for HTTP transports)
#   SCION_MCP_HOST, SCION_MCP_PORT, SCION_MCP_TRANSPORT,
#   SCION_LOG_LEVEL, SCION_SHUTDOWN_TIMEOUT,
#   SCION_METRICS_ENABLED, OPENROUTER_API_KEY, etc.
# ============================================

# ── Stage 1: Build ──────────────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency files first (layer caching — code changes don't bust this)
COPY pyproject.toml requirements.txt README.md ./

# Install Python dependencies (cached unless requirements change)
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Copy application code (separate layer — changes more often)
COPY scion/ scion/

# Install the package itself
RUN pip install --no-cache-dir --prefix=/install --no-deps .

# ── Stage 2: Runtime ────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

LABEL org.opencontainers.image.title="Scion" \
      org.opencontainers.image.description="Self-Evolving AI Skill Engine with MCP Server" \
      org.opencontainers.image.source="https://github.com/Deepfreezechill/scion"

# Create non-root user
RUN groupadd --gid 1001 scion && \
    useradd --uid 1001 --gid scion --shell /bin/bash --create-home scion

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
WORKDIR /app
COPY --chmod=755 scion/ scion/
COPY pyproject.toml ./

# Create directories for runtime data
RUN mkdir -p /app/skills /app/logs && \
    chown -R scion:scion /app

# Switch to non-root user
USER scion

# Default environment
ENV SCION_MCP_HOST=0.0.0.0 \
    SCION_MCP_PORT=8000 \
    SCION_MCP_TRANSPORT=streamable-http \
    SCION_LOG_LEVEL=INFO \
    SCION_SHUTDOWN_TIMEOUT=8 \
    SCION_METRICS_ENABLED=true \
    SCION_SKILL_STORE_PATH=/app/skills \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Health check: hits the real /health endpoint (unauthenticated)
# Default transport is streamable-http, so HTTP check works.
# For stdio transport, override with HEALTHCHECK NONE in docker-compose.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Graceful shutdown: Docker sends SIGTERM, handler drains in-flight tasks
STOPSIGNAL SIGTERM

ENTRYPOINT ["python", "-m", "scion.mcp_server"]
CMD ["--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8000"]
