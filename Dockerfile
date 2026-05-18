# ── Stage 1: Install dependencies ─────────────────────────────────────────────
FROM python:3.13-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Lean runtime image ───────────────────────────────────────────────
FROM python:3.13-slim AS runtime
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application files
COPY app.py .
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/
COPY metrics.json ./metrics.json

# Create non-root user for security
RUN useradd -m -u 1001 appuser && chown -R appuser /app
USER appuser

# Expose Flask port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"

# Start with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "60", "app:app"]