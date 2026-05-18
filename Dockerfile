# Stage 1: Install dependencies
# We use a separate builder stage just to install Python packages.
# This keeps all the build tools and pip cache out of the final image,
# which makes the runtime image smaller and faster to pull.
FROM python:3.13-slim AS builder
WORKDIR /app
COPY requirements.txt .

# Install all packages into /install instead of the system Python folders.
# --no-cache-dir stops pip from saving the download cache, saving disk space.
# --prefix=/install puts everything into one folder so we can copy it cleanly
# into the runtime stage without bringing along any build-time leftovers.
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# Stage 2: Lean runtime image
# This is the image that actually runs in production. It starts fresh from
# the same base so it has none of the build clutter from Stage 1.
FROM python:3.13-slim AS runtime
WORKDIR /app

# Copy only the installed packages from the builder stage.
# Nothing else from Stage 1 comes across — no pip, no cache, no build tools.
COPY --from=builder /install /usr/local

# Copy the application source files into the image.
# Each COPY is listed separately so Docker can cache them individually —
# if only app.py changes, Docker does not re-copy the models or data folders.
COPY app.py .
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/
COPY metrics.json ./metrics.json

# Create a non-root user and transfer ownership of the app folder to them.
# Running as root inside a container is a security risk — if the process is
# ever compromised, a non-root user limits what an attacker can do on the host.
RUN useradd -m -u 1001 appuser && chown -R appuser /app
USER appuser

# Tell Docker that this container listens on port 5000.
# This does not publish the port to the host — that is done at runtime with -p.
# It serves as documentation and is used by Docker Compose and Kubernetes.
EXPOSE 5000

# Docker will run this command every 30 seconds to check the container is healthy.
# --interval=30s  : how often to run the check
# --timeout=10s   : how long to wait before declaring the check failed
# --start-period=15s : grace period after startup before failures count
# --retries=3     : how many consecutive failures before marking as unhealthy
# The check makes an HTTP request to the /health endpoint. If it returns
# anything other than HTTP 200, the check fails.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"

# Start the Flask app using gunicorn, which is a production-grade web server.
# Flask's built-in server is for development only and cannot handle real traffic.
# --bind 0.0.0.0:5000 : listen on all network interfaces inside the container
# --workers 2         : run two worker processes to handle requests in parallel
# --timeout 60        : kill and restart a worker if it takes longer than 60 seconds
# app:app             : tells gunicorn to import the 'app' object from app.py
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "60", "app:app"]