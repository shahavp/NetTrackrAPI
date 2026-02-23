# =============================================================
#  Cricket Ball Tracker API — Railway Deployment
#  CPU-only (no CUDA required)
# =============================================================
#
# Railway-specific:
#   • Binds to $PORT (Railway injects this at runtime)
#   • Single worker (YOLO model is ~300 MB in RAM)
#   • No Docker HEALTHCHECK — Railway uses railway.json instead
#   • PYTHONUNBUFFERED so logs stream to Railway dashboard

FROM python:3.11-slim

# Logs appear in Railway dashboard without buffering delay
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System deps for OpenCV + ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0t64 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (layer cached separately from code)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY main.py tracker.py ./

# YOLO model weights — must exist before building:
#   cricket_yolov8/yolov8n_cricket_cpu/weights/best.pt
COPY cricket_yolov8/ ./cricket_yolov8/

# Working directories (Railway filesystem is ephemeral, this is fine)
RUN mkdir -p /tmp/cricket_uploads /tmp/cricket_outputs

# Non-root user
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app /tmp/cricket_uploads /tmp/cricket_outputs
USER appuser

# Railway injects $PORT at runtime — CMD reads it via shell form
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 120
