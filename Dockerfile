# ============================================================================
# MCP Aggregator - Unified Context Retrieval Middleware
# ============================================================================
# Multi-stage build for smaller final image
FROM python:3.11-slim as builder

WORKDIR /build

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ============================================================================
# Final stage - minimal runtime image
# ============================================================================
FROM python:3.11-slim

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder (use correct path for --user install)
COPY --from=builder /root/.local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /root/.local/bin /usr/local/bin

# Copy application code (minimal files only)
COPY aggregator_server.py config.py mcp_client.py __init__.py ./

# Create non-root user for security
RUN useradd -m -u 1000 mcpuser && \
    chown -R mcpuser:mcpuser /app

USER mcpuser

# Expose port
EXPOSE 9003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:9003/admin/health || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Run the aggregator server with optimizations
CMD ["python", "-u", "-O", "aggregator_server.py"]

