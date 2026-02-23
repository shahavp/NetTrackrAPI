# Cricket Ball Tracker — FastAPI on Railway

Two-pass cricket ball tracking API deployed on [Railway](https://railway.app).

**Pass 1** — YOLO + ByteTrack + momentum-gated tracking + HSV recovery → raw CSV  
**Pass 2** — Bounce-first per-segment RANSAC → two-parabola fitting → perspective-scaled rendering

---

## Deploy to Railway

### 1. Prepare the repo

```bash
# Clone / init your repo
git init cricket-tracker && cd cricket-tracker

# Copy all project files into the repo root:
#   main.py, tracker.py, Dockerfile, railway.json,
#   requirements.txt, .dockerignore

# Add your YOLO model weights
mkdir -p cricket_yolov8/yolov8n_cricket_cpu/weights
cp /path/to/best.pt cricket_yolov8/yolov8n_cricket_cpu/weights/

git add -A && git commit -m "initial"
```

### 2. Push to GitHub

```bash
gh repo create cricket-tracker --private --push
```

### 3. Deploy on Railway

1. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**
2. Select your `cricket-tracker` repo
3. Railway auto-detects the `Dockerfile` and `railway.json`
4. Wait for build + deploy (~2–4 minutes, model weights are ~6 MB)
5. Go to **Settings → Networking → Generate Domain**
6. Your API is live at `https://<your-service>.railway.app`

### 4. Configure (optional)

In Railway dashboard → **Variables**, you can override:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `cricket_yolov8/yolov8n_cricket_cpu/weights/best.pt` | Path to YOLO weights inside the image |
| `MAX_UPLOAD_MB` | `100` | Max upload file size |
| `JOB_TTL_MINUTES` | `30` | Auto-delete completed jobs after N minutes |
| `ALLOWED_ORIGINS` | `https://*.railway.app,...` | CORS origins (comma-separated) |

> `PORT` is injected automatically by Railway — do not set it manually.

---

## API Reference

Base URL: `https://<your-service>.railway.app`

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/track` | Async — returns `job_id`, poll `/status/{id}` |
| `POST` | `/track/sync` | Sync — blocks, returns video directly |
| `GET` | `/status/{job_id}` | Poll progress (0–100%) |
| `GET` | `/download/{job_id}` | Download trajectory video |
| `GET` | `/download/{job_id}/csv` | Download trajectory CSV |
| `GET` | `/download/{job_id}/detection` | Download detection video |
| `DELETE` | `/jobs/{job_id}` | Clean up job artefacts |
| `GET` | `/health` | Liveness / readiness probe |

Interactive docs: `https://<your-service>.railway.app/docs`

### POST /track parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video` | file | required | Video file (.mp4, .mov, .avi, .mkv, .webm) |
| `confidence_threshold` | float | 0.2 | YOLO confidence |
| `gate_radius_px` | int | 80 | Spatial gating radius |
| `enable_hsv_recovery` | bool | true | HSV colour fallback |
| `max_misses` | int | 60 | Misses before tracker resets |

---

## Client Usage

```bash
pip install requests

# Async (recommended)
python client.py input5.mov --url https://<your-service>.railway.app

# Sync (short clips only)
python client.py input5.mov --url https://<your-service>.railway.app --sync

# Local development
python client.py input5.mov --url http://localhost:8000
```

---

## Local Development

```bash
# Option A: Docker (matches Railway environment)
docker compose up --build
# API at http://localhost:8000

# Option B: Direct
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Project Structure

```
├── main.py              # FastAPI application (production-ready)
├── tracker.py           # CricketBallTracker class
├── Dockerfile           # Railway-compatible (reads $PORT)
├── railway.json         # Railway config-as-code
├── requirements.txt     # Python dependencies
├── .dockerignore        # Keep image lean
├── docker-compose.yml   # Local development only
├── client.py            # Test client
└── cricket_yolov8/      # YOLO model weights (you provide)
    └── yolov8n_cricket_cpu/
        └── weights/
            └── best.pt
```

## Architecture Notes

- **Single worker** — YOLO model is ~300 MB in RAM. One uvicorn worker keeps memory under Railway's limits.
- **Singleton model** — loaded once at startup, shared across requests. Per-request tuneables (confidence, gate radius) are applied without reloading weights.
- **Ephemeral storage** — Railway provides 10 GB that resets on redeploy. Completed jobs are auto-deleted after `JOB_TTL_MINUTES` (default 30). Upload + output per job is typically 5–20 MB.
- **Health check** — `/health` returns 503 until the YOLO model is loaded, then 200. Railway uses this (via `railway.json`) to know when the container is ready for traffic.
- **Background processing** — async jobs run in FastAPI's thread pool. For heavy concurrent usage, deploy a Redis-backed task queue (Celery/ARQ) behind the same Railway project.
