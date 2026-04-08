# Cricket Ball Tracking API — Complete System Context

**Version:** 3.0.0
**Runtime:** Python 3.11 / Railway PaaS
**Architecture:** Distributed microservices — stateless HTTP API + async ML worker + Redis + S3-compatible object store

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Service Decomposition](#2-service-decomposition)
3. [API Service](#3-api-service)
4. [Worker Service](#4-worker-service)
5. [Supporting Infrastructure](#5-supporting-infrastructure)
6. [Tracking Engine — Scientific Dissection](#6-tracking-engine--scientific-dissection)
7. [Complete Job Lifecycle](#7-complete-job-lifecycle)
8. [API Endpoint Reference](#8-api-endpoint-reference)
9. [Data Schemas](#9-data-schemas)
10. [Storage Architecture](#10-storage-architecture)
11. [Containerisation](#11-containerisation)
12. [Configuration & Environment Variables](#12-configuration--environment-variables)

---

## 1. Architecture Overview

The system is split into five Railway services that communicate exclusively through Redis and an S3-compatible object store. No service calls another service directly over HTTP.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Client (mobile app / CLI)                                          │
│    POST /track  →  multipart video upload                           │
│    GET  /status/{job_id}  →  poll progress + presigned URLs         │
└────────────────────┬────────────────────────────────────────────────┘
                     │ HTTPS
┌────────────────────▼────────────────────────────────────────────────┐
│  API Service  (Dockerfile.api, 2 replicas)                          │
│  FastAPI + Uvicorn — stateless, no ML                               │
│  • Validates upload → streams to Bucket → creates Redis job → enqueue│
│  • Generates presigned S3 URLs for completed outputs                │
└──────┬─────────────────────────────────┬───────────────────────────┘
       │ Redis RPUSH                      │ boto3 upload_fileobj
┌──────▼───────────┐          ┌──────────▼──────────────────────────┐
│  Redis           │          │  Railway Bucket  (S3-compatible)     │
│  job:{id} JSON   │          │  uploads/{job_id}.mp4                │
│  TTL = 30 min    │          │  outputs/{job_id}/trajectory.mp4     │
│  "tracking" queue│          │  outputs/{job_id}/trajectory.csv     │
└──────┬───────────┘          └──────────────────────────────────────┘
       │ BLPOP
┌──────▼───────────────────────────────────────────────────────────────┐
│  Worker Service  (Dockerfile.worker, 1 replica)                      │
│  RQ worker — forks one child process per job                         │
│  • Downloads video from Bucket                                       │
│  • Runs full two-pass ML pipeline (tracker.py)                       │
│  • Uploads outputs to Bucket                                         │
│  • Writes final job state to Redis                                   │
└──────────────────────────────────────────────────────────────────────┘
                     ↑
┌────────────────────┴────────────────────────────────────────────────┐
│  Cleanup Cron  (Dockerfile.api image, hourly cron)                  │
│  cleanup.py — scans Bucket, deletes objects whose Redis job expired  │
└─────────────────────────────────────────────────────────────────────┘
```

**Key design invariants:**
- The API holds no ML model and no per-request state. It can run as many replicas as needed.
- All job state lives in Redis under `job:{job_id}` with a 30-minute TTL.
- All file I/O (upload, outputs) goes through the Railway Bucket; nothing is stored on container filesystems beyond the job's `TemporaryDirectory`.
- The worker processes one job at a time per replica, with no shared model state between jobs.

---

## 2. Service Decomposition

| Service | Railway config | Dockerfile | Replicas | Role |
|---|---|---|---|---|
| **api** | `railway.json` | `Dockerfile.api` | 2 | Stateless HTTP gateway |
| **worker** | `railway.worker.json` | `Dockerfile.worker` | 1 | ML pipeline executor |
| **redis** | Railway template | N/A | 1 | Job state + queue |
| **tracking-bucket** | Railway Bucket | N/A | N/A | S3-compatible object storage |
| **cleanup-cron** | `railway.cleanup.json` | `Dockerfile.api` | cron | Hourly bucket cleanup |

**Watch patterns** (which file changes trigger a Railway rebuild):
- API service: `requirements.api.txt`, `api.py`, `jobs.py`, `task_queue.py`, `storage.py`, `Dockerfile.api`
- Worker service: `requirements.txt`, `requirements.worker.txt`, `tracker.py`, `worker.py`, `jobs.py`, `storage.py`, `Dockerfile.worker`
- Cleanup cron: `requirements.api.txt`, `cleanup.py`, `jobs.py`, `storage.py`, `Dockerfile.api`

---

## 3. API Service

### 3.1 Runtime

- **Framework:** FastAPI 0.104+ on Uvicorn with 2 worker processes
- **Image:** `python:3.11-slim` with no ML dependencies (image ≈ 150 MB)
- **Start command:** `uvicorn api:app --host 0.0.0.0 --port ${PORT} --workers 2 --timeout-keep-alive 30`
- **Health check:** `GET /health` — responds immediately (no model to warm)
- **CORS:** None — clients are native mobile apps, not browsers

### 3.2 Dependencies (`requirements.api.txt`)

```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6   # multipart/form-data for video upload
pydantic>=2.0.0
redis>=5.0.0
rq>=1.16.0                # enqueue jobs
boto3>=1.34.0             # Railway Bucket (S3-compatible)
```

### 3.3 Module Map

| Module | Role |
|---|---|
| `api.py` | FastAPI application, all route handlers |
| `jobs.py` | Redis-backed job state CRUD |
| `task_queue.py` | RQ enqueue wrapper |
| `storage.py` | S3 client (upload, download, presigned URLs) |

### 3.4 Upload Validation

Three checks run before any storage operation:

1. **File extension** (`_validate_upload`) — must be in `{".mp4", ".avi", ".mov", ".mkv", ".webm"}` → HTTP 400 on failure
2. **File size** (`_validate_upload`) — must not exceed `MAX_UPLOAD_MB` (default 100 MB) → HTTP 413 on failure
3. **Magic-byte content check** (async, in `POST /track`) — reads the first 12 bytes of the upload stream and verifies they match a known video container signature:
   - `RIFF....` → AVI
   - `\x1a\x45\xdf\xa3` → MKV / WEBM
   - `????ftyp` / `????mdat` / `????moov` / `????free` / `????skip` → MP4 / MOV (ISO base media)
   - Mismatch → HTTP 422 (catches PDF-as-.mp4, images, etc.)
   - Empty upload (zero bytes read) → HTTP 422

After the magic-byte read the stream is rewound (`seek(0)`) so boto3 receives the complete file for S3 upload.

### 3.5 `POST /track` — Full Workflow

```
1. _validate_upload(filename, size)            → HTTP 400/413 on failure
2. await video.read(12) + seek(0)              → HTTP 422 if empty or non-video magic bytes
3. job_id = uuid4()
4. storage.upload_fileobj(video.file,
       upload_key(job_id))                     → streams to Bucket, no RAM buffer
5. jobs.create(job_id, {status:"pending", …}) → Redis SETEX with 30-min TTL
6. q.enqueue_tracking_job(job_id, params)      → Redis RPUSH on "tracking" queue
7. return _public_status(job_id, jobs.get(job_id))  → HTTP 202 JobStatus
```

The video is **never buffered in API process RAM** — the 12-byte header read is negligible, and `upload_fileobj` uses boto3's multipart streaming directly from the HTTP request body to the bucket.

### 3.6 Presigned URL Generation (`api.py:_public_status`)

Called on every `GET /status/{job_id}` and on the `POST /track` response. When `status == "completed"`, generates two 15-minute presigned S3 GET URLs:
- `trajectory.mp4` → `output_url`
- `trajectory.csv` → `csv_url`

`detection_url` is always `null` — the detection video is no longer produced (see §6.3.3).

URL generation failures are non-fatal (logged as warnings); the status response still returns with null URL fields.

---

## 4. Worker Service

### 4.1 Runtime

- **Queue consumer:** RQ (`redis-queue`) listening on `"tracking"` queue
- **Worker mode:** Default fork worker — one `os.fork()` per job
- **Start command:** `rq worker --url $REDIS_URL --with-scheduler tracking`
- **Image:** `python:3.11-slim` with full ML stack (image ≈ 2 GB)

### 4.2 Dependencies (`requirements.txt` + `requirements.worker.txt`)

```
# ML / Computer Vision
torch, torchvision          (CPU-only, from PyTorch whl index)
ultralytics>=8.0.0          (YOLO wrapper + ByteTrack)
opencv-python-headless>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
lap                         (Linear Assignment Problem — ByteTrack dependency)
onnx>=1.14.0
onnxruntime>=1.16.0         (ONNX inference runtime — used by Ultralytics)
scipy>=1.10.0               (scipy.linalg.cholesky — UKF sigma points)

# Infrastructure
redis>=5.0.0
rq>=1.16.0
boto3>=1.34.0

# API
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6
pydantic>=2.0.0
```

### 4.3 Fork Model & ONNX Safety

RQ calls `os.fork()` for every job. The child process inherits the parent's entire memory space. To prevent ONNX Runtime thread-pool corruption across fork boundaries:

1. **No module-level model preloading** — `worker.py` creates NO `CricketBallTracker` at import time. Each job's forked child loads a fresh `YOLO` instance with a fresh `InferenceSession`. Because no ONNX session is ever created in the parent process, there is no thread-pool state to corrupt across fork.

`OMP_NUM_THREADS` is intentionally **not set** — allowing ONNX Runtime to use all available CPU threads in each child process for full inference throughput.

### 4.4 `run_tracking_job(job_id, params)` — Complete Execution Flow

```python
# 1. Update status → "processing"
jobs.update(job_id, status="processing", message="Downloading input video")

with TemporaryDirectory() as tmpdir:
    video_in = Path(tmpdir) / "input.mp4"
    out_dir  = Path(tmpdir) / "output"

    # 2. Download video from Bucket
    storage.download_file(upload_key(job_id), str(video_in))

    # 3. Validate video (probe with OpenCV before running ML)
    cap = cv2.VideoCapture(str(video_in))
    assert cap.isOpened()
    fps   = cap.get(CAP_PROP_FPS)       # must be > 0
    count = cap.get(CAP_PROP_FRAME_COUNT)  # must be > 0
    cap.release()

    # 4. Build per-job TrackerConfig (fresh, no shared state)
    cfg = TrackerConfig(
        model_path=MODEL_PATH,
        conf=params.get("confidence_threshold", 0.20),
        gate_radius_px=params.get("gate_radius_px", 80),
        enable_hsv_recovery=params.get("enable_hsv_recovery", True),
        max_misses=params.get("max_misses", 60),
    )
    tracker = CricketBallTracker(config=cfg)  # loads fresh YOLO in this child process

    # 5. Run two-pass pipeline (see §6)
    result = tracker.process_video(str(video_in), str(out_dir), progress_callback=_progress)

    # 6. Upload outputs to Bucket
    _upload_outputs(job_id, out_dir)
    # Uploads: trajectory_output.mp4 → trajectory.mp4
    #          trajectory.csv → trajectory.csv

    # 7. Delete raw upload (no longer needed)
    storage.delete_object(upload_key(job_id))

    # 8. Finalise job state
    jobs.update(job_id, status="completed", progress=100.0,
                bounce=bounce,
                release_speed_kmh=_safe_float(result.get("release_speed_kmh")),
                bounce_speed_kmh=_safe_float(result.get("bounce_speed_kmh")))
```

**`_safe_float`:** Converts any `NaN` or `±inf` float to `None` before passing to `jobs.update`. Prevents degenerate IMM outputs from producing non-standard JSON tokens (`NaN`) in Redis that would cause `json.loads` to fail on the next poll.

**Bounce extraction:** Uses `if result.get("bounce") is not None:` (not a bare truth-check) to safely handle numpy arrays returned by `_smooth_trajectory`. A bare `if arr:` raises `ValueError` for multi-element numpy arrays.

**Error handling:** Any exception in steps 2–6 calls `_fail(job_id, message)` which sets `status="failed"` in Redis and re-raises, causing RQ to mark the job as failed.

**Progress reporting:** `_progress(cur, total, message)` → `jobs.update(progress=pct, message=msg)` is called on every 10th frame during both passes, making the `GET /status` response show live progress.

---

## 5. Supporting Infrastructure

### 5.1 `jobs.py` — Redis Job State

**Key format:** `job:{job_id}` → JSON string
**TTL:** 30 minutes (configurable via `JOB_TTL_SECONDS`)
**Connection:** New `Redis.from_url(REDIS_URL)` per call (no connection pooling — safe for forked workers)

| Function | Behaviour |
|---|---|
| `create(job_id, data)` | `SETEX key TTL json` — adds `_created` timestamp |
| `get(job_id)` | `GET key` → deserialised dict, or `None` if expired |
| `update(job_id, **fields)` | `GET` → merge fields → `SETEX key max(ttl, 60) json` |
| `delete(job_id)` | `DEL key` |
| `exists(job_id)` | `EXISTS key` → bool |

**TTL preservation on update:** The remaining TTL is read via `PTTL` before writing, and floored at 60 seconds. This prevents rapid successive updates from silently shrinking the job's lifetime.

**NaN safety (`_safe_dumps`):** All writes (`create`, `update`) go through `_safe_dumps` which calls `json.dumps(data, allow_nan=False)`. If that raises (because a NaN or inf float slipped through from the ML pipeline), non-finite floats are coerced to `null` and the dump is retried. This ensures Redis never stores the non-standard JSON tokens `NaN` or `Infinity`, which would cause `json.loads` to fail on the next read.

**Job state fields:**

| Field | Type | Description |
|---|---|---|
| `status` | str | `pending` → `processing` → `completed` / `failed` |
| `progress` | float | 0.0–100.0 |
| `message` | str | Human-readable phase description |
| `bounce` | dict\|null | `{x: float, y: float}` in pixel coords |
| `release_speed_kmh` | float\|null | IMM-estimated ball release speed |
| `bounce_speed_kmh` | float\|null | Always `null` — 2-mode IMM does not compute bounce speed |
| `_created` | float | Unix timestamp of job creation |

### 5.2 `task_queue.py` — RQ Enqueue Wrapper

```python
Queue("tracking", connection=Redis.from_url(REDIS_URL)).enqueue(
    "worker.run_tracking_job",
    job_id, params,
    job_id=job_id,           # RQ internal job ID = our job_id
    result_ttl=JOB_TTL,
    failure_ttl=JOB_TTL,
    job_timeout=30*60,       # max wall-time before RQ kills the job
)
```

`job_id` is used as both the application-level identifier and the RQ job ID, ensuring consistency in the RQ admin panel.

### 5.3 `storage.py` — Railway Bucket Client

All operations use a fresh `boto3.client("s3", ...)` per call (stateless, safe for multi-replica).

**Signature version:** `s3v4` — required for Railway Bucket compatibility.

**Key conventions:**

| Key pattern | Contents | Lifetime |
|---|---|---|
| `uploads/{job_id}.mp4` | Raw client upload | Deleted by worker after download |
| `outputs/{job_id}/trajectory.mp4` | Trajectory overlay video (with detection boxes) | Until job TTL expires (cleaned by cron) |
| `outputs/{job_id}/trajectory.csv` | Per-frame tracking CSV | Until job TTL expires |

**Presigned URLs:** 15-minute expiry (configurable via `PRESIGN_EXPIRES_SECONDS`). Generated only when status is `completed`.

### 5.4 `cleanup.py` — Hourly Cron

Runs as a Railway Cron service (schedule: `0 * * * *`). Exits with code 0 when complete.

**Algorithm:**
1. List all objects under `outputs/` prefix (paginated).
2. Extract `job_id` from key path `outputs/{job_id}/...`.
3. If `jobs.exists(job_id)` is `False` (Redis key expired) → delete the bucket object.
4. List all objects under `uploads/` prefix.
5. If `LastModified < now - 2*JOB_TTL` and `jobs.exists(job_id)` is `False` → delete stale upload (handles worker crashes before consumption).

The cron uses the **same Docker image as the API** (`Dockerfile.api`) since it only needs Redis and S3 access — no ML dependencies.

---

## 6. Tracking Engine — Scientific Dissection

`tracker.py` implements a two-pass pipeline. Pass 1 runs YOLO-based detection and single-ball Kalman-gated tracking. Pass 2 performs physics-based trajectory smoothing, IMM+UKF speed estimation, and visual overlay rendering.

### 6.1 `TrackerConfig` — All Parameters

| Parameter | Default | Description |
|---|---|---|
| `model_path` | `"cricket_yolov8/best.onnx"` | ONNX model weights path |
| `imgsz` | 1280 | YOLO input resolution (px) — must match training resolution |
| `conf` | 0.20 | YOLO detection confidence threshold |
| `iou` | 0.7 | YOLO NMS IoU threshold |
| `device` | `None` | Inference device (`"cpu"`, `"cuda:0"`, or auto) |
| `classes` | `[0]` | YOLO class filter — class 0 = cricket ball |
| `track_buffer` | 90 | ByteTrack frames to hold a lost track before deletion |
| `track_high_thresh` | 0.35 | ByteTrack high-confidence association threshold |
| `track_low_thresh` | 0.05 | ByteTrack low-confidence association threshold |
| `new_track_thresh` | 0.45 | ByteTrack minimum score to start a new track |
| `match_thresh` | 0.80 | ByteTrack IoU match threshold |
| `gate_radius_px` | 80 | Euclidean gate radius for candidate selection (pixels) |
| `gate_grow_per_miss` | 15 | Gate expansion per consecutive miss (pixels) |
| `max_misses` | 60 | Consecutive misses before tracker resets |
| `min_box_area_px` | 25 | Minimum detection bounding box area to accept |
| `vel_ema_alpha` | 0.3 | Velocity EMA smoothing (`new_vel` weight) |
| `enable_hsv_recovery` | `True` | Enable HSV colour blob recovery on miss |
| `hsv_h_margin` | 12 | Hue range ± margin in HSV degrees |
| `hsv_s_margin` | 70 | Saturation range ± margin |
| `hsv_v_margin` | 70 | Value range ± margin |
| `hsv_min_s` | 40 | Minimum saturation for HSV model pixels |
| `hsv_min_v` | 40 | Minimum value for HSV model pixels |
| `hsv_roi_pad_px` | 40 | HSV search ROI padding beyond gate |
| `hsv_min_area_px` | 6 | Minimum blob area to accept from HSV |
| `hsv_max_area_px` | 2000 | Maximum blob area to accept from HSV |
| `hsv_aspect_min` | 0.35 | Minimum blob aspect ratio (w/h) |
| `hsv_aspect_max` | 2.8 | Maximum blob aspect ratio |
| `hsv_morph_kernel` | 3 | Morphological kernel size for noise removal |
| `hsv_morph_iters` | 1 | Morphological open/close iterations |
| `traj_min_points` | 5 | Minimum valid detections for trajectory rendering |
| `traj_bounce_thresh` | 8 | Minimum y-pixel change to qualify as a bounce |
| `traj_angle_cutoff` | 45 | Direction change angle (deg) to cut at bat contact |
| `line_color` | `(0, 140, 255)` | Trajectory overlay colour (BGR orange) |

---

### 6.2 Pass 1 — YOLO + ByteTrack + Euclidean Gating + HSV Recovery

**Entry:** `CricketBallTracker._detection_pass(video_path, out_dir, cb)`
**Returns:** `(csv_rows: list[dict], fps: float, boxes_by_frame: dict[int, tuple])`

#### Step 1 — ByteTrack YAML generation

A custom ByteTrack configuration is written to `{out_dir}/bytetrack_custom.yaml` using the `TrackerConfig` ByteTrack parameters. This overrides Ultralytics' internal defaults.

#### Step 2 — YOLO streaming inference

```python
model.track(
    source=video_path,
    imgsz=cfg.imgsz,       # 1280 px
    conf=cfg.conf,          # 0.15
    iou=cfg.iou,            # 0.7
    classes=cfg.classes,    # [0]
    tracker=bytetrack_yaml,
    persist=True,           # ByteTrack state persists across frames
    stream=True,            # generator — one result object per frame
    verbose=False,
    save=False,
)
```

`persist=True` instructs Ultralytics to maintain ByteTrack state across successive `model.track()` calls for the same video stream. `stream=True` yields one `Results` object per frame without buffering the entire video.

#### Step 3 — Candidate extraction (per frame)

From each `Results` object:
- `result.boxes.xyxy` → float32 bounding boxes
- `result.boxes.conf` → detection confidence scores
- `result.boxes.id` → ByteTrack track IDs (integer, or None if no tracks)

Boxes smaller than `min_box_area_px = 25 px²` are discarded. Each accepted box becomes a `candidate` dict: `{xyxy, conf, id}`.

#### Step 4 — Single-ball selection via Euclidean gating

The tracker maintains one `active_id` (the ByteTrack track ID of the ball) and a predicted position `pred = (last_pos[0] + vel[0], last_pos[1] + vel[1])`.

**Selection logic:**
```
gate = gate_radius_px + misses × gate_grow_per_miss

if active_id is None:           → initialisation phase
    chosen = argmax(conf) across all candidates

else:                           → tracking phase
    pool = candidates with id == active_id, else all candidates
    if pred is available:
        best = argmin(Euclidean dist to pred) over pool
        chosen = best if dist(best, pred) ≤ gate else None
    else:
        chosen = argmax(conf) over pool
```

The gate expands linearly with consecutive misses to account for increasing position uncertainty. If `misses > max_misses`, the tracker fully resets (`active_id = None`, `last_pos = None`, `vel = 0`).

#### Step 5 — HSV colour model recovery

When `chosen is None` (YOLO missed the ball):
1. **Learning phase** (on each successful detection): crop the bounding box from the BGR frame, convert to HSV, compute median H/S/V over pixels with sufficient saturation and value. Build a hue range tolerating wrap-around (e.g. red spans 170–10°). Store as `hsv_model = {hue_ranges: [((h_lo,s_lo,v_lo), (h_hi,s_hi,v_hi)), ...]}`.
2. **Recovery phase** (on miss): define an ROI around `pred` with radius `gate + hsv_roi_pad_px`. Apply HSV thresholding, morphological open+close, and contour extraction. Filter by area (6–2000 px²) and aspect ratio (0.35–2.8). Select the contour whose centre is closest to `pred`. Return as full-frame xyxy.

Recovered detections receive `status = "hsv"` and `conf = 0.0`.

#### Step 6 — Velocity EMA update

```python
new_vel = (cx - last_pos[0], cy - last_pos[1])  # pixel displacement
vel = 0.7 × vel + 0.3 × new_vel                 # EMA with alpha=0.3
```

The EMA dampens noise from HSV blob jitter while preserving the general direction of motion for next-frame prediction.

#### Step 7 — Per-frame output

Each frame produces one CSV row and one entry in `boxes_by_frame`:

**CSV columns:** `frame, status, id, x, y, vx, vy, gate, misses`

| `status` | Meaning |
|---|---|
| `det` | YOLO detection, within gate |
| `hsv` | HSV blob recovery |
| `pred` | No detection; position recorded as predicted |
| `miss` | No detection, no prediction (before first detection) |
| `reset` | Tracker state was reset this frame |

**`boxes_by_frame`:** `{frame_index → (xyxy_tuple, hsv_recovered_bool)}` — used by Pass 2 to re-draw bounding boxes without re-reading the annotated detection video.

**CSV file** (`trajectory.csv`): always written to disk for bucket upload and Pass 2 consumption — even when zero detections occurred (headers-only in that case). Hardcoded fieldnames `["frame", "status", "id", "x", "y", "vx", "vy", "gate", "misses"]` are used when `csv_rows` is empty.

---

### 6.3 Pass 2 — IMM+UKF Speed Estimation + Parabolic Trajectory + Rendering

**Entry:** `CricketBallTracker._rendering_pass(video_path, out_dir, csv_rows, fps, boxes_by_frame, cb)`
**Returns:** `(output_vid: Path, bounce: tuple|None, release_kmh: float|None)`

#### 6.3.1 IMM+UKF Speed Estimation (`_run_imm_speed`)

This sub-pipeline operates on the in-memory `csv_rows` list. It models the ball's 3D trajectory using a 2-mode Interacting Multiple Model (IMM) filter, each mode running an Unscented Kalman Filter (UKF).

**Early exit guard:** If `csv_rows` is empty or the resulting DataFrame has no `x`/`y` columns (e.g. the YOLO loop exited immediately with zero frames), the function returns `{"imm_success": False}` immediately without running UKF. If fewer than 5 `det`/`hsv` rows remain after filtering, the same early-exit is taken.

**Camera model (`CameraModel`):**

A pinhole camera calibrated for the behind-the-stumps broadcast view:

| Parameter | Value |
|---|---|
| Focal length | 1371.0 px (fx = fy) |
| Principal point | (960, 540) px — 1080p image centre |
| Camera world position | (-2.0, 0.0, 0.71) m |
| Camera tilt | 4.19° downward (measured from pitch geometry) |

The rotation matrix `R` combines a base rotation (aligns camera axes to world axes) with a downward tilt. `project(p_world)` → pixel (u, v) and `back_project_ray(u, v)` → unit world ray are the two projection operations used throughout.

**State vector:** `x = [X, Y, Z, VX, VY, VZ]` (metres and m/s in world coordinates)

**Initial state estimation (`_init_3d_state`):**
1. Back-project the mean pixel position of the first 6 observations as a ray.
2. Place the initial 3D position where the ray intersects a nominal pitch distance (~18 m).
3. Estimate initial velocity from the pixel displacement trend of the first few observations.
4. Clip position to physically plausible pitch bounds.
5. Set initial covariance `P0 = diag([4, 1, 0.5, 50, 10, 10])`.

**UKF sigma point generation (`_sigma_points`):**

Uses the scaled unscented transform with parameters:
- `α = 1e-3` (spread of sigma points)
- `β = 2.0` (optimal for Gaussian distributions)
- `κ = 0.0`
- `λ = α²(n+κ) - n` where `n = 6`

The 13 sigma points are `x ± col_i(chol((n+λ)P))` for `i = 0..5`, with mean and covariance weights `Wm[0] = λ/(n+λ)`, `Wm[i>0] = 1/(2(n+λ))`.

**IMM modes:**

| Mode | Dynamics function | Process noise Q (diagonal) | Physical meaning |
|---|---|---|---|
| 0 — `ballistic` | `_f_ballistic` | [0.01, 0.005, 0.005, 0.5, 0.3, 0.5] | Ball in clean flight |
| 1 — `occluded` | `_f_ballistic` | [0.05, 0.02, 0.02, 2.0, 1.0, 2.0] | Ball occluded or not detected |

Both modes share the same dynamics function. The occluded mode has wider process noise, allowing the filter to coast further through gaps in detection without diverging.

**Dynamics:**
- `_f_ballistic(x, dt)`: constant velocity + gravitational acceleration (`g = 9.81 m/s²` on Z-axis)

**IMM mode probability transition matrix (2×2):**

```
              ballistic  occluded
ballistic  [   0.95       0.05  ]
occluded   [   0.15       0.85  ]
```

Initial mode probabilities: `μ = [0.92, 0.08]`.

**IMM step (`IMMFilter.step`):**

1. **Mixing:** Compute conditional mode probabilities `ω[i,j] = P[i→j]μ[i]/c[j]`. Mix state and covariance for each target mode: `x̄_j = Σᵢ ω[i,j] xᵢ`, `P̄_j = Σᵢ ω[i,j](Pᵢ + (xᵢ-x̄_j)(xᵢ-x̄_j)ᵀ)`.
2. **Predict:** Each UKF mode propagates sigma points through `_f_ballistic`.
3. **Update:** If measurement `z=[u,v]` is available (status `det` or `hsv`):
   - Measurement noise: `R = diag([σ², σ²])` where `σ = 3.0 px` (det) or `8.0 px` (hsv)
   - Project sigma points through camera: `z_sig[i] = cam.project(sig[i, :3])`
   - Compute innovation covariance S, cross-covariance Pxz, Kalman gain K
   - Mahalanobis distance check: if `δᵀS⁻¹δ > 16`, penalise likelihood by 0.01
   - Update state and covariance with Joseph symmetrisation
   - Compute likelihood: `L = exp(-½ maha) / sqrt((2π)² det(S))`
4. **No measurement:** set all likelihoods to 1.0, nudge `μ[1]` (occluded) up by ×1.3 per miss, ×1.5 when `misses_count > 5`.
5. **Mode probability update:** `μ_j ∝ L_j · c_j`
6. **Fused state:** `x̂ = Σ_j μ_j x_j`

**Speed extraction:**
- `release_speed_kmh`: median of IMM-fused 3D speed (m/s × 3.6) over first 8 frames, cleaned of outliers > 2× median, clipped to [0, 165] km/h
- `bounce_speed_kmh`: **always `null`** — the 2-mode IMM does not have a dedicated post-bounce mode

#### 6.3.2 Parabolic Trajectory Smoothing (`_load_trajectory`)

Reads `trajectory.csv` from disk. Filters to rows with `status ∈ {det, hsv, detected}` and non-null `x`, `y`. Requires at least `traj_min_points = 5` valid points.

**False-positive cleaning (`_clean_detections`):**

Before smoothing, two sequential filters are applied:

1. **Onset jump filter:** Walk forward and find the first inter-frame displacement ≥ 100 px. Everything before that jump is a static false positive (ceiling light, sight screen edge) — discard it. The ball enters from this jump's destination.

2. **Bounce-gap HSV filter:** Locate the largest frame gap between consecutive YOLO `det` frames. Any HSV-only detections falling inside that gap are scored by their implied speed relative to the last confirmed YOLO position. Points with speed outside [10, 50] px/frame are discarded — these are pitch marks, shadows, or creases that the HSV blob detector mistakenly tracks during the bounce occlusion. This filter only applies when there are ≥ 6 YOLO detections and the gap is > 5 frames.

After cleaning, fewer than `traj_min_points` remaining points raise a `ValueError` which silently skips Pass 2.

**Bounce detection (image-space, 2D):**

Scans the y-coordinate sequence (image y increases downward) for a sign reversal in `dy = diff(y)`:
- Was falling (`dy[i-1] > 0`) → now rising (`dy[i] < 0`)
- Local peak-to-peak range > `traj_bounce_thresh = 8 px`
- First such reversal is the bounce index.

**Parabolic fitting (`np.polyfit(x, y, 2)`):**

- If no bounce: fit a single degree-2 polynomial to all points
- If bounce detected: fit `x_pre → y_pre` with uniform weights; fit `x_post → y_post` with **weighted polyfit** where HSV-recovered points receive weight 2.0 and YOLO-detected points weight 1.0 — this de-emphasises noisy transition-zone YOLO detections and lets the confirmed HSV points anchor the post-bounce curve

The parabolic model is physically appropriate because a ball under gravity follows a parabolic arc in the vertical plane, and the image projection approximately preserves this shape for the field of view and distances involved.

**Cutoff detection (bat contact):**

After the bounce, scan consecutive segment pairs for a direction change exceeding `traj_angle_cutoff = 45°` (using the dot product of successive displacement vectors). Truncate the trajectory at that frame — this removes the post-bat-contact segment.

**Output:** `{points: np.int32[N,2], bounce: (x,y)|None, start_frame: int, end_frame: int}`

#### 6.3.3 Visual Rendering

The rendering loop reads directly from the **original input video** (avoiding mp4v codec read-back issues on Linux). For each frame `fn`:

1. If `fn ∈ boxes_by_frame`: draw detection bounding box
   - Green `(0, 255, 0)` = YOLO detection
   - Cyan `(0, 255, 255)` = HSV recovery
2. If `fn ≥ traj['start_frame']`: call `_draw_overlay(frame, traj, cfg.line_color, release_kmh)`
   - Orange polyline along smoothed trajectory points (`cv2.polylines`, 2px, anti-aliased)
   - Bounce marker: outer white circle (r=12), inner orange circle (r=8) at bounce pixel position
   - If `release_kmh` is available: semi-transparent black panel in top-left corner with `"Release: X km/h"` text (Helvetica 0.9 scale, white, alpha-blended at 0.55)

Output written to `trajectory_output.mp4` using mp4v codec at original video FPS and resolution.

---

### 6.4 `process_video` — Orchestration

```python
def process_video(video_path, output_dir, progress_callback=None) -> dict:

    csv_rows, fps, boxes_by_frame = self._detection_pass(video_path, out, _cb)

    traj_video = bounce = release_kmh = None
    try:
        traj_video, bounce, release_kmh = self._rendering_pass(
            video_path, out, csv_rows, fps, boxes_by_frame, _cb)
    except Exception as e:
        _cb(0, 0, f"Pass 2 skipped: {e}")   # skips rendering for any error (< 5 pts, KeyError, etc.)

    return {
        "detection_video":   str(out / "annotated_detection.mp4"),
        "trajectory_video":  str(traj_video) if traj_video else None,
        "csv":               str(out / "trajectory.csv"),
        "bounce":            bounce,
        "fps":               fps,
        "release_speed_kmh": release_kmh,
        "bounce_speed_kmh":  None,   # 2-mode IMM does not compute bounce speed
    }
```

Pass 2 failures (insufficient detections, IOError) are caught and logged as a progress message. The job still completes with `status="completed"` but with null trajectory and speed fields.

---

## 7. Complete Job Lifecycle

```
CLIENT                API (replica A or B)        REDIS          WORKER (child)       BUCKET
  │                          │                      │                 │                  │
  │─ POST /track ───────────▶│                      │                 │                  │
  │  (multipart video)       │── upload_fileobj ───────────────────────────────────────▶│
  │                          │                      │                 │                  │
  │                          │── jobs.create ──────▶│ job:{id}=pending                  │
  │                          │── q.enqueue ────────▶│ "tracking" queue += job_id        │
  │                          │                      │                 │                  │
  │◀─ 202 {job_id,pending} ──│                      │                 │                  │
  │                          │                      │                 │                  │
  │  (poll)                  │                      │◀── BLPOP ───────│                  │
  │─ GET /status/{id} ──────▶│── jobs.get ─────────▶│                 │                  │
  │◀─ {processing, 0%} ──────│                      │                 │                  │
  │                          │                      │                 │                  │
  │                          │                      │                 │── download ──────▶│
  │                          │                      │                 │◀─ input.mp4 ──────│
  │                          │                      │                 │                  │
  │                          │                      │                 │  [Pass 1: YOLO]   │
  │                          │                      │◀── update 20% ──│                  │
  │                          │                      │◀── update 50% ──│                  │
  │  (poll)                  │                      │                 │                  │
  │─ GET /status/{id} ──────▶│── jobs.get ─────────▶│                 │                  │
  │◀─ {processing, 50%} ─────│                      │                 │                  │
  │                          │                      │                 │                  │
  │                          │                      │                 │  [Pass 2: Render] │
  │                          │                      │◀── update 80% ──│                  │
  │                          │                      │                 │── upload ────────▶│
  │                          │                      │                 │   trajectory.mp4  │
  │                          │                      │                 │   detection.mp4   │
  │                          │                      │                 │   trajectory.csv  │
  │                          │                      │                 │── delete upload ──▶│
  │                          │                      │◀── completed ───│                  │
  │  (poll)                  │                      │                 │                  │
  │─ GET /status/{id} ──────▶│── jobs.get ─────────▶│                 │                  │
  │                          │── presign_url ───────────────────────────────────────────▶│
  │◀─ {completed, urls} ─────│                      │                 │                  │
  │                          │                      │                 │                  │
  │─ GET /download/{id} ────▶│                      │                 │                  │
  │◀─ 302 → presigned URL ───│                      │                 │                  │
  │─ GET presigned URL ──────────────────────────────────────────────────────────────────▶│
  │◀─ video bytes ────────────────────────────────────────────────────────────────────────│
  │                          │                      │                 │                  │
  │─ DELETE /jobs/{id} ─────▶│── delete bucket ────────────────────────────────────────▶│
  │                          │── jobs.delete ──────▶│ DEL job:{id}                      │
  │◀─ {detail: deleted} ─────│                      │                 │                  │
```

**Job state machine:**

```
pending → processing → completed
                    ↘ failed
```

**Expiry (without explicit DELETE):**
- After 30 minutes: Redis key `job:{id}` expires → `GET /status` returns 404
- Next hourly cron run: bucket objects under `outputs/{id}/` are deleted

---

## 8. API Endpoint Reference

### `POST /track`

Upload a video for asynchronous tracking.

**Request:** `multipart/form-data`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `video` | file | Yes | — | Video file (.mp4, .avi, .mov, .mkv, .webm), max 100 MB |
| `confidence_threshold` | float | No | 0.20 | YOLO detection confidence |
| `gate_radius_px` | int | No | 80 | Euclidean gate radius (pixels) |
| `enable_hsv_recovery` | bool | No | true | Enable HSV blob recovery on YOLO miss |
| `max_misses` | int | No | 60 | Reset threshold for consecutive misses |

**Response:** HTTP 202 `JobStatus`

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "progress": 0.0,
  "message": "Queued",
  "output_url": null,
  "csv_url": null,
  "detection_url": null,
  "bounce": null,
  "release_speed_kmh": null,
  "bounce_speed_kmh": null
}
```

**Error responses:**
- `400` — unsupported file extension
- `413` — file exceeds `MAX_UPLOAD_MB`
- `422` — empty file, or file content does not match a supported video container
- `500` — S3 upload failure

---

### `GET /status/{job_id}`

Poll job progress. When `status == "completed"`, includes 15-minute presigned download URLs.

**Response:** `JobStatus`

```json
{
  "job_id": "550e8400-...",
  "status": "completed",
  "progress": 100.0,
  "message": "Processing complete",
  "output_url": "https://bucket.railway.app/outputs/.../trajectory.mp4?X-Amz-...",
  "csv_url": "https://bucket.railway.app/outputs/.../trajectory.csv?X-Amz-...",
  "detection_url": null,
  "bounce": {"x": 342.0, "y": 618.0},
  "release_speed_kmh": 128.4,
  "bounce_speed_kmh": null
}
```

**Error:** `404` — job not found or TTL expired

---

### `GET /download/{job_id}`

Redirect (302) to a presigned URL for `trajectory_output.mp4`.

**Precondition:** `status == "completed"` and object exists in bucket
**Error:** `400` if not completed, `404` if job or object not found

---

### `GET /download/{job_id}/csv`

Redirect (302) to presigned URL for `trajectory.csv`.

---

### `GET /download/{job_id}/detection`

**Deprecated — always returns 404.** The detection video (`annotated_detection.mp4`) is no longer produced; detection bounding boxes are drawn directly onto `trajectory_output.mp4` via `boxes_by_frame`. This endpoint remains in the router for backwards compatibility but will never find an object in the bucket.

---

### `DELETE /jobs/{job_id}`

Delete job state from Redis and all associated bucket objects.

**Deleted objects:** `trajectory.mp4`, `trajectory.csv`, `uploads/{job_id}.mp4`

**Response:** `{"detail": "deleted"}`
**Error:** `404` — job not found

---

### `GET /health`

Liveness + readiness probe. Returns immediately (no model to load).

**Response:** `{"status": "healthy", "version": "3.0.0"}`

---

### `GET /`

Service discovery — returns version and endpoint index.

---

## 9. Data Schemas

### `JobStatus` (Pydantic model)

```python
class JobStatus(BaseModel):
    job_id: str
    status: str                    # pending | processing | completed | failed
    progress: Optional[float]      # 0.0 – 100.0
    message: Optional[str]         # human-readable phase
    output_url: Optional[str]      # presigned trajectory.mp4 URL
    csv_url: Optional[str]         # presigned trajectory.csv URL
    detection_url: Optional[str]   # always null — detection video no longer produced
    bounce: Optional[dict]         # {x: float, y: float} in pixels
    release_speed_kmh: Optional[float]
    bounce_speed_kmh: Optional[float]
```

### `trajectory.csv` Schema

| Column | Type | Description |
|---|---|---|
| `frame` | int | Zero-indexed frame number |
| `status` | str | `det`, `hsv`, `pred`, `miss`, `reset` |
| `id` | int\|empty | ByteTrack track ID |
| `x` | float\|empty | Ball centre x (pixels) |
| `y` | float\|empty | Ball centre y (pixels) |
| `vx` | float | Velocity EMA x (px/frame) |
| `vy` | float | Velocity EMA y (px/frame) |
| `gate` | float | Gate radius used this frame (pixels) |
| `misses` | int | Consecutive miss count at this frame |

---

## 10. Storage Architecture

### Redis key space

```
job:{uuid4}  →  JSON, TTL=1800s
```

Internal RQ keys (managed by RQ, not the application):
```
rq:queue:tracking        (list of pending job IDs)
rq:job:{uuid4}           (RQ job metadata)
rq:finished:{uuid4}      (completed job results)
rq:failed:{uuid4}        (failed job records)
```

### S3 bucket key space

```
uploads/
  {job_id}.mp4           ← raw client upload, transient

outputs/
  {job_id}/
    trajectory.mp4       ← trajectory overlay + detection boxes (boxes_by_frame)
    trajectory.csv       ← per-frame tracking data
```

---

## 11. Containerisation

### API Image (`Dockerfile.api`)

```
Base:       python:3.11-slim
Deps:       fastapi, uvicorn, pydantic, redis, rq, boto3
Code:       api.py, jobs.py, task_queue.py, storage.py, cleanup.py
User:       appuser (uid 1000, non-root)
Cmd:        uvicorn api:app --host 0.0.0.0 --port ${PORT} --workers 2 ...
Size:       ≈ 150 MB
```

### Worker Image (`Dockerfile.worker`)

```
Base:       python:3.11-slim
System:     libgl1, libglib2.0-0t64, libsm6, libxext6, libxrender-dev,
            libgomp1, ffmpeg
Deps:       torch (CPU), torchvision, ultralytics, opencv-python-headless,
            numpy, pandas, lap, onnx, onnxruntime, scipy, redis, rq, boto3
Code:       tracker.py, worker.py, jobs.py, storage.py
Model:      cricket_yolov8/best.onnx (baked into image)
User:       appuser (uid 1000, non-root)
Cmd:        rq worker --url $REDIS_URL --with-scheduler tracking
Size:       ≈ 2 GB
```

**ONNX fork safety:** No `InferenceSession` is created in the parent process (no module-level model preload in `worker.py`). Each forked child creates its own session from scratch, so there is no thread-pool state to corrupt. `OMP_NUM_THREADS` is left unset so ONNX Runtime uses all available CPU threads for full inference throughput.

---

## 12. Configuration & Environment Variables

### API service

| Variable | Source | Default | Description |
|---|---|---|---|
| `REDIS_URL` | Railway reference | — | Redis connection URL |
| `BUCKET_NAME` | Railway reference | — | S3 bucket name |
| `BUCKET_ENDPOINT` | Railway reference | — | S3 endpoint URL |
| `BUCKET_ACCESS_KEY_ID` | Railway reference | — | S3 access key |
| `BUCKET_SECRET_ACCESS_KEY` | Railway reference | — | S3 secret key |
| `BUCKET_REGION` | Railway reference | `auto` | S3 region |
| `PORT` | Railway injected | 8000 | Uvicorn bind port |
| `MAX_UPLOAD_MB` | manual | 100 | Maximum video upload size |
| `PRESIGN_EXPIRES_SECONDS` | manual | 900 | Presigned URL validity (15 min) |
| `JOB_TTL_SECONDS` | manual | 1800 | Redis job TTL + RQ result TTL (30 min) |

### Worker service

| Variable | Source | Default | Description |
|---|---|---|---|
| `REDIS_URL` | Railway reference | — | Redis connection URL |
| `BUCKET_*` | Railway reference | — | Same as API |
| `MODEL_PATH` | manual | `cricket_yolov8/best.onnx` | ONNX model path inside container |
| `WORKER_JOB_TIMEOUT` | manual | 1800 | Max seconds RQ allows per job |
| `JOB_TTL_SECONDS` | manual | 1800 | RQ result TTL |

### Railway reference variable syntax

```
REDIS_URL=${{Redis.REDIS_URL}}
BUCKET_NAME=${{tracking-bucket.BUCKET}}
BUCKET_ENDPOINT=${{tracking-bucket.ENDPOINT}}
BUCKET_ACCESS_KEY_ID=${{tracking-bucket.ACCESS_KEY_ID}}
BUCKET_SECRET_ACCESS_KEY=${{tracking-bucket.SECRET_ACCESS_KEY}}
BUCKET_REGION=${{tracking-bucket.REGION}}
```

---

## Appendix — `client.py` Usage

`client.py` is a standalone CLI reference client for testing the API.

```bash
# Against Railway deployment
python client.py input.mov --url https://your-service.railway.app

# Against local dev stack
python client.py input.mov --url http://localhost:8000

# Tune parameters
python client.py input.mov --conf 0.15 --gate 80 --max-misses 60 --no-hsv
```

The client: uploads video → polls status every 2 s → downloads all three outputs on completion → calls `DELETE /jobs/{id}` to clean up.

---

*This document describes the complete state of the API as of the current codebase. Standalone files (`hsv_imm_pipeline.py`, `trajectorywriter.py`, `main.py`) are not part of the production API and are excluded.*
