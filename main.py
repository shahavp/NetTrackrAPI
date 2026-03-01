"""
Cricket Ball Tracking API — Railway Production Deployment
=========================================================

FastAPI service wrapping the two-pass cricket ball tracker.

All tracking parameters (YOLO confidence, gating radius, HSV recovery,
ByteTrack thresholds, trajectory fitting constants) are locked to their
tested defaults defined in tracker.py's TrackerConfig dataclass.  The API
surface is intentionally minimal: upload a video, get results.

Railway-specific:
    • Reads $PORT from environment (Railway injects this)
    • Pre-loads YOLO model at startup so health check is meaningful
    • Auto-cleans ephemeral storage (Railway gives 10 GB, lost on redeploy)
    • Single shared tracker instance (YOLO model is ~300 MB in RAM)
    • Concurrency semaphore prevents parallel inference from causing OOM
    • Structured logging visible in Railway dashboard

Endpoints:
    POST /track              — async processing (returns job_id)
    GET  /status/{id}        — poll job progress
    GET  /download/{id}      — download trajectory video
    GET  /download/{id}/csv  — download raw trajectory CSV
    GET  /download/{id}/detection — download detection-only video
    POST /track/sync         — blocking (returns video directly)
    GET  /health             — liveness / readiness probe
    DELETE /jobs/{id}        — manual cleanup

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import asyncio
import logging
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# IMPORTANT: We do NOT import tracker.py at module level.
#
#     from tracker import CricketBallTracker, TrackerConfig   ← DON'T DO THIS
#
# That single line triggers the entire heavyweight import chain:
#     tracker.py → ultralytics → torch + torchvision → ~500 MB of RAM
#     tracker.py → cv2, numpy, pandas                → ~100 MB more
#
# All of this happens BEFORE uvicorn can bind the port.  On Railway's
# Trial plan (512 MB) it OOM-kills the process instantly.  Even on the
# Hobby plan (8 GB), it blocks for 30-60 seconds of dead silence where
# Railway's healthcheck gets connection-refused.
#
# Instead, we import tracker.py lazily inside _load_model() — which runs
# in a background thread AFTER uvicorn has already bound the port and
# started serving /health responses.


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
# NOTE: Default must match the path baked into the Docker image by the
# Dockerfile's COPY step (cricket_yolov8/ → ./cricket_yolov8/).  The
# original default "cricket_yolov8/best.pt" was wrong — it didn't
# account for the nested yolov8n_cricket_cpu/weights/ subdirectory,
# causing a FileNotFoundError at startup every time.
MODEL_PATH       = os.getenv("MODEL_PATH", "cricket_yolov8/yolov8n_cricket_cpu/weights/best.pt")
UPLOAD_DIR       = Path(os.getenv("UPLOAD_DIR", "/tmp/cricket_uploads"))
OUTPUT_DIR       = Path(os.getenv("OUTPUT_DIR", "/tmp/cricket_outputs"))
MAX_UPLOAD_MB    = int(os.getenv("MAX_UPLOAD_MB", "100"))
JOB_TTL_MINUTES  = int(os.getenv("JOB_TTL_MINUTES", "30"))
ALLOWED_ORIGINS  = os.getenv(
    "ALLOWED_ORIGINS",
    "https://*.railway.app,http://localhost:3000,http://localhost:8000"
).split(",")

# Accepted video file extensions — anything outside this set is rejected
# immediately to avoid wasting time uploading unsupported formats.
VALID_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ---------------------------------------------------------------------------
#  Shared state
# ---------------------------------------------------------------------------
JOBS: dict[str, dict] = {}
_tracker = None   # CricketBallTracker singleton, loaded in background thread
_model_ready = False
_model_error: str | None = None              # set if model loading crashes

# Concurrency gate: only 1 video can be processed at a time.
# YOLO inference + OpenCV frame buffers consume ~1-3 GB of RAM per job.
# Running two jobs in parallel would exceed Railway's memory limits and
# cause an OOM kill.  The semaphore queues any additional jobs until the
# current one finishes, which is safer than letting them run side-by-side.
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "1"))
_processing_semaphore: asyncio.Semaphore | None = None  # initialised in lifespan


# ---------------------------------------------------------------------------
#  Pydantic response model
# ---------------------------------------------------------------------------

class JobStatus(BaseModel):
    """Schema returned by /track and /status — everything a client needs
    to track progress and locate the finished artefacts."""
    job_id: str
    status: str                          # pending | processing | completed | failed
    progress: Optional[float] = None
    message: Optional[str] = None
    output_url: Optional[str] = None
    csv_url: Optional[str] = None
    detection_url: Optional[str] = None
    bounce: Optional[dict] = None


# ---------------------------------------------------------------------------
#  Auto-cleanup (Railway ephemeral storage is 10 GB)
# ---------------------------------------------------------------------------

def _cleanup_loop():
    """Background thread: delete completed/failed jobs older than JOB_TTL_MINUTES.

    Runs in an infinite loop with a 60-second sleep between sweeps.
    Railway's filesystem is ephemeral (wiped on redeploy), but within a
    single deployment window old jobs would accumulate without this.
    """
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
    """Remove all on-disk artefacts and the in-memory record for a job."""
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
    """Startup / shutdown hook for the FastAPI application.

    CRITICAL DESIGN DECISION — model loading happens AFTER yield:

        The YOLO model takes 30–120 seconds to load on Railway's shared
        CPU.  If we load it before yield, uvicorn won't bind the port
        until loading finishes.  During that window Railway's healthcheck
        gets connection-refused (not even a 503), which counts as a
        failure — and if it exceeds startupTimeout the deploy is killed.

        By yielding first, uvicorn binds the port immediately.  The
        /health endpoint returns 503 ("Model still loading") which tells
        Railway the container is alive but not ready.  Once the background
        thread finishes loading, /health flips to 200.

    Startup sequence:
        1. Create upload and output directories
        2. Initialise the concurrency semaphore (must be inside the loop)
        3. yield — uvicorn binds the port, Railway can reach /health
        4. Background thread loads the YOLO model
        5. Background cleanup thread starts

    Shutdown:
        Log and exit — Railway handles container teardown.
    """
    global _processing_semaphore

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create the semaphore here, inside the running event loop.
    # asyncio.Semaphore must be instantiated in an async context so it
    # attaches to the correct loop — creating it at module level would
    # bind to no loop and raise RuntimeError on Python 3.10+.
    _processing_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
    log.info("Concurrency limit set to %d simultaneous job(s)", MAX_CONCURRENT_JOBS)

    # Spawn the model loader in a daemon thread so it runs AFTER yield.
    # This lets uvicorn bind the port first, making /health reachable
    # while the heavy YOLO import + weight load happens in parallel.
    model_thread = threading.Thread(target=_load_model, daemon=True)
    model_thread.start()

    # Start the periodic cleanup daemon
    cleanup_thread = threading.Thread(target=_cleanup_loop, daemon=True)
    cleanup_thread.start()

    # Yield control — uvicorn binds the port NOW, Railway can healthcheck
    yield

    log.info("Shutting down")


def _load_model():
    """Load the YOLO model in a background thread.

    Runs after uvicorn has bound the port so that /health is reachable
    during loading.  This function performs the LAZY IMPORT of tracker.py
    (and therefore ultralytics, torch, cv2, numpy, pandas) — all ~600 MB
    of dependencies load here, not at module import time.

    Any exception is caught and stored in _model_error so the health
    endpoint can report exactly what went wrong, rather than the container
    silently crashing with no diagnostics.
    """
    global _tracker, _model_ready, _model_error

    try:
        log.info("Loading YOLO model from %s ...", MODEL_PATH)

        # Verify the weights file actually exists before attempting to
        # load — gives a clear error message instead of a cryptic
        # ultralytics traceback.
        weights = Path(MODEL_PATH)
        if not weights.exists():
            raise FileNotFoundError(
                f"Model weights not found at '{MODEL_PATH}'. "
                f"Contents of parent dir: {list(weights.parent.iterdir()) if weights.parent.exists() else 'PARENT DIR MISSING'}"
            )

        # LAZY IMPORT — this is where the heavy dependencies actually load.
        # Importing tracker pulls in ultralytics → torch → ~500 MB.
        # Doing it here (in a background thread, after yield) means the
        # port is already bound and /health is already serving 503s.
        from tracker import CricketBallTracker, TrackerConfig

        _tracker = CricketBallTracker(config=TrackerConfig(model_path=MODEL_PATH))
        _tracker.warmup()
        _model_ready = True
        log.info("Model ready — accepting requests")

    except Exception as e:
        # Store the error so /health can report it.  The container stays
        # alive (Railway won't get connection-refused), but /health will
        # keep returning 503 with the error message, which is visible
        # in the Railway dashboard logs.
        _model_error = str(e)
        log.error("Failed to load model: %s", e, exc_info=True)


app = FastAPI(
    title="Cricket Ball Tracking API",
    description=(
        "Upload a cricket video and receive an annotated trajectory overlay.\n\n"
        "Two-pass pipeline: YOLO + ByteTrack detection with HSV recovery, "
        "then bounce-aware two-parabola fitting with perspective-scaled rendering.\n\n"
        "All detection and tracking parameters use tested defaults — the API "
        "accepts only a video file."
    ),
    version="2.1.0",
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
    """Reject unsupported file types and oversized uploads early.

    Called before any bytes are written to disk so we fail fast on
    obviously invalid requests without consuming storage or bandwidth.
    """
    ext = Path(filename).suffix.lower()
    if ext not in VALID_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported format '{ext}'. Accepted: {', '.join(sorted(VALID_EXTENSIONS))}",
        )
    # size can be 0 for chunked uploads — the streaming check in
    # _save_upload catches those.  We still reject known-oversized files.
    if size > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            413,
            f"File too large ({size / 1024 / 1024:.0f} MB). Max: {MAX_UPLOAD_MB} MB",
        )


def _job_output_dir(job_id: str) -> Path:
    """Return the output directory path for a given job."""
    return OUTPUT_DIR / job_id


def _progress_cb(job_id: str):
    """Return a callback that updates job progress in the JOBS dict.

    The tracker calls this periodically with (current_frame, total_frames,
    message) so the /status endpoint can report live progress.
    """
    def _cb(current: int, total: int, message: str):
        pct = round(100 * current / max(total, 1), 1)
        JOBS[job_id]["progress"] = pct
        JOBS[job_id]["message"] = message
    return _cb


def _get_tracker():
    """Return the singleton tracker instance.

    The YOLO model is loaded once during startup (in the background
    _load_model thread) and reused for every request.  TrackerConfig
    defaults are baked in — no per-request parameter changes.

    The fallback path (tracker is None) should only trigger during
    testing or if _load_model somehow didn't run.  It lazy-imports
    tracker.py since we don't import it at module level.
    """
    global _tracker
    if _tracker is None:
        # Fallback in case _load_model didn't run (e.g. during testing).
        # Lazy import here too — tracker.py pulls in torch/ultralytics.
        from tracker import CricketBallTracker, TrackerConfig
        _tracker = CricketBallTracker(config=TrackerConfig(model_path=MODEL_PATH))
        _tracker.warmup()
    return _tracker


def _run_pipeline(job_id: str, video_path: Path):
    """Background worker — runs the full two-pass tracker pipeline.

    This is a synchronous, blocking function designed to be called inside
    asyncio.to_thread so the event loop stays responsive while YOLO and
    OpenCV crunch frames.

    Updates the JOBS dict in-place with status, progress, and results.
    """
    try:
        JOBS[job_id]["status"] = "processing"
        JOBS[job_id]["message"] = "Initialising tracker"
        log.info("Job %s: processing %s", job_id, video_path.name)

        tracker = _get_tracker()
        out_dir = _job_output_dir(job_id)
        result = tracker.process_video(
            video_path=str(video_path),
            output_dir=str(out_dir),
            progress_callback=_progress_cb(job_id),
        )

        # Mark the job as complete and populate download URLs
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["progress"] = 100.0
        JOBS[job_id]["message"] = "Processing complete"
        JOBS[job_id]["output_url"] = f"/download/{job_id}"
        JOBS[job_id]["csv_url"] = f"/download/{job_id}/csv"
        JOBS[job_id]["detection_url"] = f"/download/{job_id}/detection"

        # Store bounce coordinates if the tracker detected one
        if result.get("bounce"):
            bx, by = result["bounce"]
            JOBS[job_id]["bounce"] = {"x": bx, "y": by}

        log.info("Job %s: completed", job_id)

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["message"] = f"Failed: {e}"
        log.error("Job %s: failed — %s", job_id, e, exc_info=True)


# Semaphore-gated wrapper for the pipeline.
# This is the function that asyncio.create_task actually runs.  It acquires
# the semaphore before offloading the heavy synchronous work to a thread,
# ensuring only MAX_CONCURRENT_JOBS run at once.  Any excess jobs wait
# in the queue with a "Waiting" status until a slot opens up.
async def _run_pipeline_gated(job_id: str, video_path: Path):
    """Acquire the processing semaphore, then run the pipeline in a thread."""
    JOBS[job_id]["message"] = "Waiting for available processing slot"
    log.info("Job %s: waiting for semaphore (%d max concurrent)",
             job_id, MAX_CONCURRENT_JOBS)

    async with _processing_semaphore:
        # asyncio.to_thread runs _run_pipeline in the default executor
        # (thread pool) so it doesn't block the event loop while YOLO
        # crunches frames.  The semaphore stays held until the thread
        # finishes, preventing a second job from starting in parallel.
        await asyncio.to_thread(_run_pipeline, job_id, video_path)


async def _save_upload(video: UploadFile, video_path: Path):
    """Stream an uploaded video to disk in 1 MB chunks.

    Returns the total number of bytes written.  Raises HTTPException(413)
    if the stream exceeds MAX_UPLOAD_MB — this catches chunked uploads
    where Content-Length was missing or incorrect.
    """
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

def _require_model_ready():
    """Reject the request early if the YOLO model isn't loaded yet.

    Called at the top of /track and /track/sync so we don't accept a
    video upload only to fail when the pipeline tries to run inference.
    Returns a clear 503 with the loading status or failure reason.
    """
    if not _model_ready:
        if _model_error:
            raise HTTPException(503, f"Model failed to load: {_model_error}")
        raise HTTPException(503, "Model still loading, try again shortly")

@app.post("/track", response_model=JobStatus)
async def track_async(
    video: UploadFile = File(..., description="Video file (.mp4, .mov, .avi, .mkv, .webm)"),
):
    """Upload a video for **asynchronous** tracking.

    Returns a ``job_id`` immediately.  Poll ``/status/{job_id}`` to check
    progress, then download results from ``/download/{job_id}``.

    All detection and tracking parameters use tested defaults.
    """
    _require_model_ready()
    _validate_upload(video.filename, video.size or 0)

    job_id = str(uuid.uuid4())
    video_path = UPLOAD_DIR / f"{job_id}.mp4"
    await _save_upload(video, video_path)

    # Initialise the job record before spawning the background task
    JOBS[job_id] = {
        "status": "pending", "progress": 0.0,
        "message": "Queued", "output_url": None,
        "csv_url": None, "detection_url": None, "bounce": None,
        "_created": time.time(),
    }

    # Use asyncio.create_task instead of BackgroundTasks so the async
    # semaphore wrapper (_run_pipeline_gated) is awaited properly.
    # BackgroundTasks.add_task would call a sync function directly in the
    # thread pool, bypassing our semaphore entirely.
    asyncio.create_task(_run_pipeline_gated(job_id, video_path))

    log.info("Job %s: queued (%s)", job_id, video.filename)
    return JobStatus(
        job_id=job_id,
        **{k: v for k, v in JOBS[job_id].items() if not k.startswith("_")},
    )


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    """Check processing progress for a given job."""
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found")
    return JobStatus(
        job_id=job_id,
        **{k: v for k, v in JOBS[job_id].items() if not k.startswith("_")},
    )


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
    video: UploadFile = File(..., description="Video file (.mp4, .mov, .avi, .mkv, .webm)"),
):
    """**Synchronous** tracking — blocks until complete, returns the video.

    Best for short clips (< 30 seconds).  For longer videos use the async
    ``/track`` endpoint and poll ``/status/{job_id}`` instead.

    This endpoint respects the concurrency semaphore, so it will wait if
    another job is already processing.  This prevents two YOLO inference
    runs from overlapping and causing an OOM kill on Railway.

    All detection and tracking parameters use tested defaults.
    """
    _require_model_ready()
    _validate_upload(video.filename, video.size or 0)
    job_id = str(uuid.uuid4())
    video_path = UPLOAD_DIR / f"{job_id}.mp4"

    try:
        await _save_upload(video, video_path)

        # Acquire the semaphore so this sync job doesn't overlap with
        # any async jobs currently processing.  The actual inference is
        # offloaded to a thread to avoid blocking the event loop.
        async with _processing_semaphore:
            await asyncio.to_thread(
                _run_sync_pipeline, video_path, job_id
            )

        out_dir = _job_output_dir(job_id)
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
        # Always clean up the uploaded file — output dir is left for the
        # FileResponse to serve, then auto-cleaned by _cleanup_loop.
        if video_path.exists():
            video_path.unlink(missing_ok=True)


def _run_sync_pipeline(video_path: Path, job_id: str):
    """Thread-safe synchronous pipeline runner for the /track/sync endpoint.

    Separated into its own function so asyncio.to_thread can invoke it
    cleanly.  This keeps the actual YOLO + OpenCV work off the event loop.
    """
    tracker = _get_tracker()
    out_dir = _job_output_dir(job_id)
    tracker.process_video(str(video_path), str(out_dir))


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and all its artefacts."""
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found")
    _cleanup_job(job_id)
    return {"detail": "deleted"}


@app.get("/health")
async def health():
    """Liveness + readiness probe.

    Railway hits this path (configured in railway.json) to decide whether
    the container is ready to receive traffic.  Returns 503 until the YOLO
    model has finished loading.
    """
    if not _model_ready:
        # Distinguish between "still loading" and "loading crashed" so
        # the Railway dashboard logs tell the operator exactly what's
        # happening.  Both return 503 (not ready), but the message
        # makes debugging much faster.
        if _model_error:
            raise HTTPException(503, f"Model failed to load: {_model_error}")
        raise HTTPException(503, "Model still loading")

    # Calculate how many processing slots are currently free.
    # _value is the internal counter on asyncio.Semaphore — when it's 0,
    # all slots are occupied and new jobs will queue.
    slots_free = _processing_semaphore._value if _processing_semaphore else 0

    return {
        "status": "healthy",
        "model_loaded": True,
        "active_jobs": sum(1 for j in JOBS.values() if j["status"] == "processing"),
        "queued_jobs": sum(1 for j in JOBS.values() if j["status"] == "pending"),
        "total_jobs": len(JOBS),
        "max_concurrent": MAX_CONCURRENT_JOBS,
        "slots_available": slots_free,
    }


@app.get("/")
async def root():
    """Service info and endpoint index."""
    return {
        "service": "Cricket Ball Tracking API",
        "version": "2.1.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "POST /track": "Async tracking (upload video, returns job_id)",
            "POST /track/sync": "Sync tracking (upload video, returns video)",
            "GET /status/{job_id}": "Poll progress",
            "GET /download/{job_id}": "Trajectory video",
            "GET /download/{job_id}/csv": "Trajectory CSV",
            "GET /download/{job_id}/detection": "Detection video",
            "DELETE /jobs/{job_id}": "Cleanup artefacts",
            "GET /health": "Liveness / readiness probe",
        },
    }