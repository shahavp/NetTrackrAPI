#!/usr/bin/env python3
"""
Cricket Ball Tracker — API Client
==================================

Usage:
    # Against Railway deployment
    python client.py input5.mov --url https://your-service.railway.app

    # Local development
    python client.py input5.mov

    # Sync mode (short clips, blocks until done)
    python client.py input5.mov --sync

    # Tune parameters
    python client.py input5.mov --conf 0.3 --gate 100 --no-hsv
"""

import argparse
import sys
import time
from pathlib import Path

import requests


def track_async(url: str, video_path: str, params: dict) -> None:
    """Submit video, poll for completion, download all outputs."""
    print(f"Uploading {video_path} ...")
    with open(video_path, "rb") as f:
        resp = requests.post(
            f"{url}/track",
            files={"video": (Path(video_path).name, f, "video/mp4")},
            data=params,
            timeout=120,
        )
    resp.raise_for_status()
    job = resp.json()
    job_id = job["job_id"]
    print(f"Job queued: {job_id}")

    while True:
        time.sleep(2)
        r = requests.get(f"{url}/status/{job_id}", timeout=10)
        r.raise_for_status()
        s = r.json()
        status = s["status"]
        pct = s.get("progress") or 0
        msg = s.get("message", "")
        print(f"  [{status}] {pct:.1f}% — {msg}        ", end="\r", flush=True)

        if status == "completed":
            print()
            break
        if status == "failed":
            print(f"\nFailed: {msg}")
            sys.exit(1)

    # Download outputs
    stem = Path(video_path).stem
    for suffix, endpoint, label in [
        ("_trajectory.mp4", f"/download/{job_id}", "Trajectory video"),
        ("_detection.mp4",  f"/download/{job_id}/detection", "Detection video"),
        ("_trajectory.csv", f"/download/{job_id}/csv", "Trajectory CSV"),
    ]:
        out = f"{stem}{suffix}"
        print(f"  Downloading {label} -> {out}")
        r = requests.get(f"{url}{endpoint}", timeout=120)
        if r.status_code == 200:
            with open(out, "wb") as f:
                f.write(r.content)
            print(f"    {len(r.content) / 1024:.0f} KB")
        else:
            print(f"    (not available: {r.status_code})")

    if s.get("bounce"):
        b = s["bounce"]
        print(f"\nBounce at ({b['x']:.0f}, {b['y']:.0f}) px")

    # Clean up server-side
    requests.delete(f"{url}/jobs/{job_id}", timeout=10)
    print("Done.")


def track_sync(url: str, video_path: str, params: dict) -> None:
    """Blocking upload — returns video directly."""
    print(f"Uploading {video_path} (sync, will block) ...")
    with open(video_path, "rb") as f:
        resp = requests.post(
            f"{url}/track/sync",
            files={"video": (Path(video_path).name, f, "video/mp4")},
            data=params,
            timeout=600,
        )
    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}")
        sys.exit(1)

    out = f"{Path(video_path).stem}_tracked.mp4"
    with open(out, "wb") as f:
        f.write(resp.content)
    print(f"Saved: {out} ({len(resp.content) / 1024:.0f} KB)")


def main():
    p = argparse.ArgumentParser(description="Cricket Tracker API Client")
    p.add_argument("video", help="Path to input video")
    p.add_argument("--url", default="http://localhost:8000",
                   help="API base URL (e.g. https://your-service.railway.app)")
    p.add_argument("--sync", action="store_true", help="Blocking sync mode")
    p.add_argument("--conf", type=float, default=0.2, help="Confidence threshold")
    p.add_argument("--gate", type=int, default=80, help="Gate radius (px)")
    p.add_argument("--no-hsv", action="store_true", help="Disable HSV recovery")
    p.add_argument("--max-misses", type=int, default=60)
    args = p.parse_args()

    if not Path(args.video).exists():
        print(f"File not found: {args.video}"); sys.exit(1)

    # Health check
    try:
        r = requests.get(f"{args.url}/health", timeout=10)
        if r.status_code == 503:
            print("Server is starting up (model loading). Waiting ...")
            for _ in range(60):
                time.sleep(5)
                r = requests.get(f"{args.url}/health", timeout=10)
                if r.status_code == 200:
                    break
            else:
                print("Server did not become ready in time."); sys.exit(1)
        r.raise_for_status()
        info = r.json()
        print(f"Server: {info.get('status')} | "
              f"Active jobs: {info.get('active_jobs', '?')}")
    except requests.ConnectionError:
        print(f"Cannot reach {args.url}"); sys.exit(1)

    params = {
        "confidence_threshold": args.conf,
        "gate_radius_px": args.gate,
        "enable_hsv_recovery": not args.no_hsv,
        "max_misses": args.max_misses,
    }

    if args.sync:
        track_sync(args.url, args.video, params)
    else:
        track_async(args.url, args.video, params)


if __name__ == "__main__":
    main()
