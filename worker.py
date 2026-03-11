"""
RQ worker — cricket ball tracking job runner.

Start command (Railway worker service):
    rq worker --url $REDIS_URL tracking

How it works
------------
* RQ forks a child process for every job (default fork worker).
* The YOLO model is loaded once at module level in the parent process.
  Each forked child inherits it via copy-on-write, so model weights are
  not reloaded from disk for every job — only on cold start.
* Each job gets its own TrackerConfig instance, so per-request parameters
  (confidence_threshold, gate_radius_px, …) never mutate shared state.
* All file I/O goes through /tmp inside a TemporaryDirectory that is
  automatically cleaned up when the job exits, whether it succeeds or fails.

Environment variables
---------------------
  REDIS_URL     — injected by Railway reference variable
  MODEL_PATH    — path to ONNX / PT weights inside the container image
  BUCKET_*      — injected by Railway bucket reference variables
"""

import logging
import os
import tempfile
from pathlib import Path

import jobs
import storage
from tracker import CricketBallTracker, TrackerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("cricket-worker")

MODEL_PATH = os.getenv("MODEL_PATH", "cricket_yolov8/best.onnx")

# ---------------------------------------------------------------------------
#  Module-level model load (runs once in the parent worker process).
#  RQ's fork worker inherits this via copy-on-write, so every child gets
#  the model without paying the disk-load cost again.
# ---------------------------------------------------------------------------
log.info("Loading YOLO model from %s …", MODEL_PATH)
_base_tracker = CricketBallTracker(config=TrackerConfig(model_path=MODEL_PATH))
_base_tracker.warmup()
log.info("Model ready — worker is listening")


# ---------------------------------------------------------------------------
#  Job function (registered with RQ via dotted import "worker.run_tracking_job")
# ---------------------------------------------------------------------------

def run_tracking_job(job_id: str, params: dict) -> dict:
    """Process one cricket tracking job end-to-end.

    Called by the RQ worker in a forked child process.
    Returns a summary dict (stored as the RQ job result in Redis).
    """
    log.info("Job %s: starting", job_id)
    jobs.update(job_id, status="processing", message="Downloading input video")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp       = Path(tmpdir)
        video_in  = tmp / "input.mp4"
        out_dir   = tmp / "output"
        out_dir.mkdir()

        # ── Download input from bucket ────────────────────────────────────
        try:
            storage.download_file(storage.upload_key(job_id), str(video_in))
        except Exception as exc:
            _fail(job_id, f"Failed to download input: {exc}")
            raise

        # ── Build a per-job config (immutable params, shared model weights) ─
        cfg = TrackerConfig(
            model_path=MODEL_PATH,
            conf=params.get("confidence_threshold", 0.2),
            gate_radius_px=params.get("gate_radius_px", 80),
            enable_hsv_recovery=params.get("enable_hsv_recovery", True),
            max_misses=params.get("max_misses", 60),
        )
        # Reuse the already-loaded YOLO model; avoid paying disk I/O again.
        tracker = CricketBallTracker(config=cfg)
        tracker._model = _base_tracker._model   # share loaded weights (read-only in child)

        jobs.update(job_id, message="Initialising tracker")

        # ── Progress callback ─────────────────────────────────────────────
        def _progress(cur: int, total: int, message: str):
            pct = round(100 * cur / max(total, 1), 1)
            jobs.update(job_id, progress=pct, message=message)

        # ── Run the two-pass pipeline ─────────────────────────────────────
        try:
            result = tracker.process_video(
                str(video_in), str(out_dir), progress_callback=_progress
            )
        except Exception as exc:
            _fail(job_id, f"Pipeline error: {exc}")
            raise

        # ── Upload outputs to bucket ──────────────────────────────────────
        jobs.update(job_id, message="Uploading outputs")
        _upload_outputs(job_id, out_dir)

        # ── Delete raw upload (no longer needed) ──────────────────────────
        try:
            storage.delete_object(storage.upload_key(job_id))
        except Exception:
            pass   # not fatal; cleanup cron will catch stragglers

        # ── Finalise job state ────────────────────────────────────────────
        bounce = None
        if result.get("bounce"):
            bx, by = result["bounce"]
            bounce = {"x": float(bx), "y": float(by)}

        jobs.update(
            job_id,
            status="completed",
            progress=100.0,
            message="Processing complete",
            bounce=bounce,
            release_speed_kmh=result.get("release_speed_kmh"),
            bounce_speed_kmh=result.get("bounce_speed_kmh"),
        )
        log.info("Job %s: complete", job_id)
        return {"job_id": job_id, "bounce": bounce}


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

_OUTPUT_MAP = [
    ("trajectory_output.mp4", "trajectory.mp4"),
    ("annotated_detection.mp4", "detection.mp4"),
    ("trajectory.csv", "trajectory.csv"),
]


def _upload_outputs(job_id: str, out_dir: Path) -> None:
    for local_name, bucket_suffix in _OUTPUT_MAP:
        local = out_dir / local_name
        if local.exists():
            key = storage.output_key(job_id, bucket_suffix)
            storage.upload_file(str(local), key)
            log.info("Job %s: uploaded %s → %s", job_id, local_name, key)
        else:
            log.warning("Job %s: expected output %s not found", job_id, local_name)


def _fail(job_id: str, message: str) -> None:
    log.error("Job %s: %s", job_id, message)
    jobs.update(job_id, status="failed", message=message)
