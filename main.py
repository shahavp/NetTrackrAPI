"""
Cricket Ball Tracking API — Railway Production Deployment
=========================================================

FastAPI service wrapping the two-pass cricket ball tracker.

Railway-specific:
    • Reads $PORT from environment (Railway injects this)
    • Pre-loads YOLO model at startup so health check is meaningful
    • Auto-cleans ephemeral storage (Railway gives 10 GB, lost on redeploy)
    • Single shared tracker instance (YOLO model is ~300 MB in RAM)
    • Structured logging visible in Railway dashboard

Endpoints:
    POST /track          — async processing (returns job_id)
    GET  /status/{id}    — poll job progress
    GET  /download/{id}  — download trajectory video
    GET  /download/{id}/csv — download raw trajectory CSV
    GET  /download/{id}/detection — download detection-only video
    POST /track/sync     — blocking (returns video directly)
    GET  /health         — liveness / readiness probe
    DELETE /jobs/{id}    — manual cleanup

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import logging
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from tracker import CricketBallTracker, TrackerConfig

# ---------------------------------------------------------------------------
#  Logging (Railway captures stdout/stderr)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("cricket-api")

# ---------------------------------------------------------------------------
#  Config from environment
# ---------------------------------------------------------------------------
MODEL_PATH       = os.getenv("MODEL_PATH", "cricket_yolov8/best.pt")
UPLOAD_DIR       = Path(os.getenv("UPLOAD_DIR", "/tmp/cricket_uploads"))
OUTPUT_DIR       = Path(os.getenv("OUTPUT_DIR", "/tmp/cricket_outputs"))
MAX_UPLOAD_MB    = int(os.getenv("MAX_UPLOAD_MB", "100"))
JOB_TTL_MINUTES  = int(os.getenv("JOB_TTL_MINUTES", "30"))
ALLOWED_ORIGINS  = os.getenv(
    "ALLOWED_ORIGINS",
    "https://*.railway.app,http://localhost:3000,http://localhost:8000"
).split(",")

VALID_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# ---------------------------------------------------------------------------
#  Shared state
# ---------------------------------------------------------------------------
JOBS: dict[str, dict] = {}
_tracker: CricketBallTracker | None = None   # singleton, loaded at startup
_model_ready = False


# ---------------------------------------------------------------------------
#  Pydantic models
# ---------------------------------------------------------------------------

class JobStatus(BaseModel):
    job_id: str
    status: str                          # pending | processing | completed | failed
    progress: Optional[float] = None
    message: Optional[str] = None
    output_url: Optional[str] = None
    csv_url: Optional[str] = None
    detection_url: Optional[str] = None
    bounce: Optional[dict] = None


class TrackingParams(BaseModel):
    """User-tuneable subset of TrackerConfig."""
    confidence_threshold: float = Field(default=0.2, ge=0.0, le=1.0,
                                        description="YOLO confidence threshold")
    gate_radius_px: int = Field(default=80, ge=10, le=500,
                                description="Base gating radius in pixels")
    enable_hsv_recovery: bool = Field(default=True,
                                      description="Enable HSV colour fallback")
    max_misses: int = Field(default=60, ge=1, le=300,
                            description="Frames before tracker resets")


# ---------------------------------------------------------------------------
#  Auto-cleanup (Railway ephemeral storage is 10 GB)
# ---------------------------------------------------------------------------

def _cleanup_loop():
    """Background thread: delete completed/failed jobs older than JOB_TTL_MINUTES."""
    while True:
        time.sleep(60)
        now = time.time()
        expired = [
            jid for jid, j in JOBS.items()
            if j.get("status") in ("completed", "failed")
            and now - j.get("_created", now) > JOB_TTL_MINUTES * 60
        ]
        for jid in expired:
            _cleanup_job(jid)
            log.info("Auto-cleaned expired job %s", jid)


def _cleanup_job(job_id: str):
    d = _job_output_dir(job_id)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    up = UPLOAD_DIR / f"{job_id}.mp4"
    if up.exists():
        up.unlink(missing_ok=True)
    JOBS.pop(job_id, None)


# ---------------------------------------------------------------------------
#  App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _tracker, _model_ready

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-load model so the first request is not slow and health check
    # correctly reports readiness.
    log.info("Loading YOLO model from %s ...", MODEL_PATH)
    _tracker = CricketBallTracker(config=TrackerConfig(model_path=MODEL_PATH))
    _tracker.warmup()
    _model_ready = True
    log.info("Model ready — accepting requests")

    # Start cleanup daemon
    t = threading.Thread(target=_cleanup_loop, daemon=True)
    t.start()

    yield

    log.info("Shutting down")


app = FastAPI(
    title="Cricket Ball Tracking API",
    description=(
        "Upload a cricket video and receive an annotated trajectory overlay.\n\n"
        "Two-pass pipeline: YOLO + ByteTrack detection with HSV recovery, "
        "then bounce-aware two-parabola fitting with perspective-scaled rendering."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _validate_upload(filename: str, size: int):
    ext = Path(filename).suffix.lower()
    if ext not in VALID_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported format '{ext}'. Accepted: {', '.join(sorted(VALID_EXTENSIONS))}",
        )
    if size > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            413,
            f"File too large ({size / 1024 / 1024:.0f} MB). Max: {MAX_UPLOAD_MB} MB",
        )


def _make_config(params: TrackingParams) -> TrackerConfig:
    return TrackerConfig(
        model_path=MODEL_PATH,
        conf=params.confidence_threshold,
        gate_radius_px=params.gate_radius_px,
        enable_hsv_recovery=params.enable_hsv_recovery,
        max_misses=params.max_misses,
    )


def _job_output_dir(job_id: str) -> Path:
    return OUTPUT_DIR / job_id


def _progress_cb(job_id: str):
    def _cb(current: int, total: int, message: str):
        pct = round(100 * current / max(total, 1), 1)
        JOBS[job_id]["progress"] = pct
        JOBS[job_id]["message"] = message
    return _cb


def _get_tracker(params: TrackingParams) -> CricketBallTracker:
    """Return the singleton tracker, optionally updating tuneable params."""
    global _tracker
    if _tracker is None:
        _tracker = CricketBallTracker(config=_make_config(params))
        _tracker.warmup()
    else:
        # Update per-request tuneables without reloading the YOLO model
        cfg = _make_config(params)
        _tracker.cfg = cfg
    return _tracker


def _run_pipeline(job_id: str, video_path: Path, params: TrackingParams):
    """Background worker — runs the full tracker pipeline."""
    try:
        JOBS[job_id]["status"] = "processing"
        JOBS[job_id]["message"] = "Initialising tracker"
        log.info("Job %s: processing %s", job_id, video_path.name)

        tracker = _get_tracker(params)
        out_dir = _job_output_dir(job_id)
        result = tracker.process_video(
            video_path=str(video_path),
            output_dir=str(out_dir),
            progress_callback=_progress_cb(job_id),
        )

        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["progress"] = 100.0
        JOBS[job_id]["message"] = "Processing complete"
        JOBS[job_id]["output_url"] = f"/download/{job_id}"
        JOBS[job_id]["csv_url"] = f"/download/{job_id}/csv"
        JOBS[job_id]["detection_url"] = f"/download/{job_id}/detection"

        if result.get("bounce"):
            bx, by = result["bounce"]
            JOBS[job_id]["bounce"] = {"x": bx, "y": by}

        log.info("Job %s: completed", job_id)

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["message"] = f"Failed: {e}"
        log.error("Job %s: failed — %s", job_id, e, exc_info=True)


async def _save_upload(video: UploadFile, video_path: Path):
    """Stream upload to disk and return final size."""
    total = 0
    with open(video_path, "wb") as f:
        while chunk := await video.read(1024 * 1024):  # 1 MB chunks
            total += len(chunk)
            if total > MAX_UPLOAD_MB * 1024 * 1024:
                f.close()
                video_path.unlink(missing_ok=True)
                raise HTTPException(413, f"Upload exceeds {MAX_UPLOAD_MB} MB limit")
            f.write(chunk)
    return total


# ---------------------------------------------------------------------------
#  Endpoints
# ---------------------------------------------------------------------------

@app.post("/track", response_model=JobStatus)
async def track_async(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    confidence_threshold: float = 0.2,
    gate_radius_px: int = 80,
    enable_hsv_recovery: bool = True,
    max_misses: int = 60,
):
    """Upload a video for **asynchronous** tracking.  Poll ``/status/{job_id}``."""
    _validate_upload(video.filename, video.size or 0)

    job_id = str(uuid.uuid4())
    video_path = UPLOAD_DIR / f"{job_id}.mp4"
    await _save_upload(video, video_path)

    params = TrackingParams(
        confidence_threshold=confidence_threshold,
        gate_radius_px=gate_radius_px,
        enable_hsv_recovery=enable_hsv_recovery,
        max_misses=max_misses,
    )

    JOBS[job_id] = {
        "status": "pending", "progress": 0.0,
        "message": "Queued", "output_url": None,
        "csv_url": None, "detection_url": None, "bounce": None,
        "_created": time.time(),
    }
    background_tasks.add_task(_run_pipeline, job_id, video_path, params)
    log.info("Job %s: queued (%s)", job_id, video.filename)
    return JobStatus(job_id=job_id, **{k: v for k, v in JOBS[job_id].items() if not k.startswith("_")})


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    """Check processing progress."""
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found")
    return JobStatus(job_id=job_id, **{k: v for k, v in JOBS[job_id].items() if not k.startswith("_")})


@app.get("/download/{job_id}")
async def download_trajectory(job_id: str):
    """Download the trajectory-overlay video."""
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found")
    if JOBS[job_id]["status"] != "completed":
        raise HTTPException(400, f"Not ready: {JOBS[job_id]['status']}")
    p = _job_output_dir(job_id) / "trajectory_output.mp4"
    if not p.exists():
        raise HTTPException(404, "Trajectory video not found")
    return FileResponse(str(p), media_type="video/mp4",
                        filename=f"trajectory_{job_id}.mp4")


@app.get("/download/{job_id}/csv")
async def download_csv(job_id: str):
    """Download the raw per-frame trajectory CSV."""
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found")
    if JOBS[job_id]["status"] != "completed":
        raise HTTPException(400, f"Not ready: {JOBS[job_id]['status']}")
    p = _job_output_dir(job_id) / "trajectory.csv"
    if not p.exists():
        raise HTTPException(404, "CSV not found")
    return FileResponse(str(p), media_type="text/csv",
                        filename=f"trajectory_{job_id}.csv")


@app.get("/download/{job_id}/detection")
async def download_detection(job_id: str):
    """Download the detection-only annotated video (bounding boxes)."""
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found")
    if JOBS[job_id]["status"] != "completed":
        raise HTTPException(400, f"Not ready: {JOBS[job_id]['status']}")
    p = _job_output_dir(job_id) / "annotated_detection.mp4"
    if not p.exists():
        raise HTTPException(404, "Detection video not found")
    return FileResponse(str(p), media_type="video/mp4",
                        filename=f"detection_{job_id}.mp4")


@app.post("/track/sync")
async def track_sync(
    video: UploadFile = File(...),
    confidence_threshold: float = 0.2,
    gate_radius_px: int = 80,
    enable_hsv_recovery: bool = True,
    max_misses: int = 60,
):
    """
    **Synchronous** tracking — blocks until complete, returns the video.

    Use ``/track`` for videos longer than ~30 seconds.
    """
    _validate_upload(video.filename, video.size or 0)
    job_id = str(uuid.uuid4())
    video_path = UPLOAD_DIR / f"{job_id}.mp4"

    try:
        await _save_upload(video, video_path)

        params = TrackingParams(
            confidence_threshold=confidence_threshold,
            gate_radius_px=gate_radius_px,
            enable_hsv_recovery=enable_hsv_recovery,
            max_misses=max_misses,
        )
        tracker = _get_tracker(params)
        out_dir = _job_output_dir(job_id)
        tracker.process_video(str(video_path), str(out_dir))

        out = out_dir / "trajectory_output.mp4"
        if not out.exists():
            out = out_dir / "annotated_detection.mp4"
        if not out.exists():
            raise HTTPException(500, "No output video produced")

        return FileResponse(str(out), media_type="video/mp4",
                            filename=f"tracked_{video.filename}")
    except HTTPException:
        raise
    except Exception as e:
        log.error("Sync job %s failed: %s", job_id, e, exc_info=True)
        raise HTTPException(500, f"Processing failed: {e}")
    finally:
        if video_path.exists():
            video_path.unlink(missing_ok=True)


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and all its artefacts."""
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found")
    _cleanup_job(job_id)
    return {"detail": "deleted"}


@app.get("/health")
async def health():
    """
    Liveness + readiness probe.

    Railway hits this path (configured in railway.json) to decide whether
    the container is ready to receive traffic.  Returns 503 until the YOLO
    model has finished loading.
    """
    if not _model_ready:
        raise HTTPException(503, "Model still loading")
    return {
        "status": "healthy",
        "model_loaded": True,
        "active_jobs": sum(1 for j in JOBS.values() if j["status"] == "processing"),
        "total_jobs": len(JOBS),
    }


@app.get("/")
async def root():
    return {
        "service": "Cricket Ball Tracking API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "POST /track": "Async tracking (returns job_id)",
            "POST /track/sync": "Sync tracking (returns video)",
            "GET /status/{job_id}": "Poll progress",
            "GET /download/{job_id}": "Trajectory video",
            "GET /download/{job_id}/csv": "Trajectory CSV",
            "GET /download/{job_id}/detection": "Detection video",
            "DELETE /jobs/{job_id}": "Cleanup artefacts",
            "GET /health": "Liveness / readiness probe",
        },
    }
