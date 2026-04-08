"""
Cricket Ball Tracking API — stateless production service
=========================================================

This module replaces main.py for multi-replica Railway deployments.

Key differences from main.py
------------------------------
* No model loading — the API is thin and fast; workers do all ML work.
* Job state lives in Redis (jobs.py), not a process-local dict.
* Uploads are streamed to the Railway Bucket via server-side PUT;
  downloads redirect to presigned bucket URLs, never proxied through the API.
* /track/sync is removed — every request becomes a queued job.
* No CORS middleware — all clients are native mobile apps, not browsers.

Endpoints
---------
  POST /track                  async tracking  → returns job_id
  GET  /status/{job_id}        poll progress
  GET  /download/{job_id}      redirect → presigned trajectory video URL
  GET  /download/{job_id}/csv  redirect → presigned CSV URL
  GET  /download/{job_id}/detection  redirect → presigned detection video URL
  DELETE /jobs/{job_id}        delete job state + bucket objects
  GET  /health                 liveness probe (no model to wait for)
  GET  /                       service info + endpoint list

Environment variables (set via Railway reference variables)
-----------------------------------------------------------
  REDIS_URL
  BUCKET_NAME, BUCKET_ENDPOINT, BUCKET_ACCESS_KEY_ID,
  BUCKET_SECRET_ACCESS_KEY, BUCKET_REGION
  MAX_UPLOAD_MB  (default 100)
"""

import logging
import os
import uuid
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

import jobs
import task_queue as q
import storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("cricket-api")

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
VALID_EXT     = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Video magic-byte signatures for content-type validation
_VIDEO_MAGIC = (
    b"RIFF",              # AVI
    b"\x1a\x45\xdf\xa3",  # MKV / WEBM
)


def _is_video_bytes(header: bytes) -> bool:
    """Return True if the first bytes look like a video container."""
    if header[:4] in _VIDEO_MAGIC:
        return True
    # MP4 / MOV: ISO box with 'ftyp', 'mdat', 'moov', 'free', or 'skip' type at offset 4
    if len(header) >= 8 and header[4:8] in (b"ftyp", b"mdat", b"moov", b"free", b"skip"):
        return True
    return False

# ---------------------------------------------------------------------------
#  Pydantic models
# ---------------------------------------------------------------------------

class JobStatus(BaseModel):
    job_id: str
    status: str                          # pending | processing | completed | failed
    progress: Optional[float] = None
    message: Optional[str]   = None
    output_url: Optional[str] = None     # presigned URL when completed
    csv_url: Optional[str]    = None
    detection_url: Optional[str] = None
    bounce: Optional[dict]    = None
    release_speed_kmh: Optional[float] = None
    bounce_speed_kmh: Optional[float]  = None


# ---------------------------------------------------------------------------
#  App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Cricket Ball Tracking API",
    description=(
        "Upload a cricket video and receive an annotated trajectory overlay.\n\n"
        "Two-pass pipeline: YOLO + ByteTrack detection with HSV recovery, "
        "then bounce-aware two-parabola fitting with perspective-scaled rendering.\n\n"
        "All heavy processing runs in separate worker services; the API is stateless."
    ),
    version="3.0.0",
)

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _validate_upload(filename: str, size: int):
    import pathlib
    ext = pathlib.Path(filename).suffix.lower()
    if ext not in VALID_EXT:
        raise HTTPException(400, f"Unsupported format '{ext}'. Accepted: {', '.join(sorted(VALID_EXT))}")
    if size > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large ({size / 1024 / 1024:.0f} MB). Max: {MAX_UPLOAD_MB} MB")


def _public_status(job_id: str, job: dict) -> JobStatus:
    """Strip internal fields and add presigned download URLs if completed."""
    output_url = csv_url = detection_url = None
    if job.get("status") == "completed":
        try:
            output_url = storage.presigned_download_url(storage.output_key(job_id, "trajectory.mp4"))
            csv_url    = storage.presigned_download_url(storage.output_key(job_id, "trajectory.csv"))
        except Exception as exc:
            log.warning("Job %s: could not generate presigned URLs — %s", job_id, exc)

    return JobStatus(
        job_id=job_id,
        status=job.get("status", "unknown"),
        progress=job.get("progress"),
        message=job.get("message"),
        bounce=job.get("bounce"),
        output_url=output_url,
        csv_url=csv_url,
        detection_url=detection_url,
        release_speed_kmh=job.get("release_speed_kmh"),
        bounce_speed_kmh=job.get("bounce_speed_kmh"),
    )


# ---------------------------------------------------------------------------
#  Endpoints
# ---------------------------------------------------------------------------

@app.post("/track", response_model=JobStatus, status_code=202)
async def track_async(
    video: UploadFile = File(...),
    confidence_threshold: float = 0.15,
    gate_radius_px: int = 80,
    enable_hsv_recovery: bool = True,
    max_misses: int = 60,
):
    """Upload a video for **asynchronous** tracking.

    Returns a ``job_id``.  Poll ``GET /status/{job_id}`` for progress.
    Download URLs are included in the status response once processing completes.
    """
    _validate_upload(video.filename or "upload.mp4", video.size or 0)

    # Read magic bytes to catch empty files and non-video content (e.g. PDF renamed .mp4)
    header = await video.read(12)
    await video.seek(0)
    if not header:
        raise HTTPException(422, "Uploaded file is empty")
    if not _is_video_bytes(header):
        raise HTTPException(422, "File content does not appear to be a supported video format")

    job_id = str(uuid.uuid4())

    # Stream upload to bucket (server-side; avoids loading entire file into RAM)
    log.info("Job %s: uploading %s to bucket", job_id, video.filename)
    try:
        storage.upload_fileobj(video.file, storage.upload_key(job_id))
    except Exception as exc:
        raise HTTPException(500, f"Upload to storage failed: {exc}") from exc

    params = {
        "confidence_threshold": confidence_threshold,
        "gate_radius_px": gate_radius_px,
        "enable_hsv_recovery": enable_hsv_recovery,
        "max_misses": max_misses,
    }

    # Persist initial job state in Redis before enqueueing so /status never 404s
    jobs.create(job_id, {
        "status": "pending",
        "progress": 0.0,
        "message": "Queued",
        "bounce": None,
    })

    q.enqueue_tracking_job(job_id, params)
    log.info("Job %s: queued (%s)", job_id, video.filename)

    job = jobs.get(job_id) or {}
    return _public_status(job_id, job)


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    """Check processing progress and retrieve download URLs when complete."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found or expired")
    return _public_status(job_id, job)


@app.get("/download/{job_id}", include_in_schema=False)
async def download_trajectory(job_id: str):
    """Redirect to a presigned URL for the trajectory overlay video."""
    return _presigned_redirect(job_id, "trajectory.mp4")


@app.get("/download/{job_id}/csv", include_in_schema=False)
async def download_csv(job_id: str):
    """Redirect to a presigned URL for the trajectory CSV."""
    return _presigned_redirect(job_id, "trajectory.csv")


@app.get("/download/{job_id}/detection", include_in_schema=False)
async def download_detection(job_id: str):
    """Redirect to a presigned URL for the detection-annotated video."""
    return _presigned_redirect(job_id, "detection.mp4")


def _presigned_redirect(job_id: str, filename: str) -> RedirectResponse:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found or expired")
    if job.get("status") != "completed":
        raise HTTPException(400, f"Not ready: {job.get('status')}")
    key = storage.output_key(job_id, filename)
    if not storage.object_exists(key):
        raise HTTPException(404, f"{filename} not found in storage")
    url = storage.presigned_download_url(key)
    return RedirectResponse(url, status_code=302)


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete job state from Redis and all associated bucket objects."""
    if not jobs.exists(job_id):
        raise HTTPException(404, "Job not found")

    # Delete bucket objects (best-effort; don't fail the request if missing)
    for filename in ("trajectory.mp4", "detection.mp4", "trajectory.csv"):
        try:
            storage.delete_object(storage.output_key(job_id, filename))
        except Exception:
            pass
    try:
        storage.delete_object(storage.upload_key(job_id))
    except Exception:
        pass

    jobs.delete(job_id)
    return {"detail": "deleted"}


@app.get("/health")
async def health():
    """Liveness + readiness probe.

    The API service has no model to load, so this returns 200 immediately.
    Worker readiness is inferred from queue depth / job progress.
    """
    return {"status": "healthy", "version": "3.0.0"}


@app.get("/")
async def root():
    return {
        "service": "Cricket Ball Tracking API",
        "version": "3.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "POST /track": "Async tracking (returns job_id, HTTP 202)",
            "GET /status/{job_id}": "Poll progress + download URLs",
            "GET /download/{job_id}": "Redirect → presigned trajectory video",
            "GET /download/{job_id}/csv": "Redirect → presigned CSV",
            "GET /download/{job_id}/detection": "Redirect → presigned detection video",
            "DELETE /jobs/{job_id}": "Delete job + bucket artefacts",
            "GET /health": "Liveness / readiness probe",
        },
    }
