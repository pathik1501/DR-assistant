# Use Python 3.10 slim image
FROM python:3.10-slim

# Cache-busting arg (change value to invalidate build cache when needed)
ARG BUILDKIT_FLUSH=1

# Set working directory
WORKDIR /app

# Install minimal system dependencies (only what's needed)
# Note: opencv-python-headless doesn't require GUI libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Ensure only headless OpenCV is present
RUN pip uninstall -y opencv-python opencv-contrib-python || true \
    && pip install --no-cache-dir --force-reinstall opencv-python-headless==4.10.0.84

# Debug: show installed cv2 and OpenCV packages
RUN python -c "import cv2, sys; print('cv2 version:', cv2.__version__); print('cv2 file:', cv2.__file__)" \
    && pip freeze | grep -i opencv || true

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models logs outputs data/vector_db

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command (can be overridden)
CMD ["python", "-m", "uvicorn", "src.inference:app", "--host", "0.0.0.0", "--port", "8080"]
