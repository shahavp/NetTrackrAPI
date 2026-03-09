"""
Cleanup cron — deletes expired bucket objects and orphaned uploads.

Deploy as a Railway Cron service pointing at this script.
Schedule: "0 * * * *" (hourly) or "*/15 * * * *" (every 15 min) as needed.

Start command for the Railway cron service:
    python cleanup.py

The script exits with code 0 when done; Railway cron expects the process
to exit (not run forever).

Why this is needed
------------------
Railway Buckets do not yet support lifecycle / expiry rules, so expired
artefacts must be removed explicitly.  Redis job TTLs handle state expiry
automatically; this script handles bucket objects.

Logic
-----
1. Scan outputs/{job_id}/* — if the corresponding Redis key is missing
   (i.e. the job TTL has expired), delete the bucket objects.
2. Scan uploads/{job_id}.mp4 — delete any raw upload older than 2× JOB_TTL
   that has no live job entry (stale from a crashed worker).

Environment variables
---------------------
  REDIS_URL
  BUCKET_NAME, BUCKET_ENDPOINT, BUCKET_ACCESS_KEY_ID,
  BUCKET_SECRET_ACCESS_KEY, BUCKET_REGION
  JOB_TTL_SECONDS  (default 1800)
"""

import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("cricket-cleanup")

JOB_TTL = int(os.getenv("JOB_TTL_SECONDS", str(30 * 60)))


def main() -> int:
    # Late imports so the script fails fast with a clear error if deps missing
    import storage
    import jobs

    now = time.time()
    deleted = 0

    # ── Clean expired output objects ──────────────────────────────────────
    log.info("Scanning outputs/ for orphaned objects …")
    for obj in storage.list_objects("outputs/"):
        key = obj["Key"]                      # e.g. "outputs/<job_id>/trajectory.mp4"
        parts = key.split("/")
        if len(parts) < 3:
            continue
        job_id = parts[1]
        if not jobs.exists(job_id):
            try:
                storage.delete_object(key)
                log.info("Deleted orphaned output: %s", key)
                deleted += 1
            except Exception as exc:
                log.warning("Failed to delete %s: %s", key, exc)

    # ── Clean stale uploads (worker crashed before consuming) ─────────────
    log.info("Scanning uploads/ for stale raw uploads …")
    stale_threshold = now - JOB_TTL * 2
    for obj in storage.list_objects("uploads/"):
        key = obj["Key"]                      # e.g. "uploads/<job_id>.mp4"
        last_modified = obj.get("LastModified")
        if last_modified is None:
            continue
        age = last_modified.timestamp()
        parts = key.split("/")
        if len(parts) < 2:
            continue
        job_id = parts[1].removesuffix(".mp4")
        if age < stale_threshold and not jobs.exists(job_id):
            try:
                storage.delete_object(key)
                log.info("Deleted stale upload: %s", key)
                deleted += 1
            except Exception as exc:
                log.warning("Failed to delete %s: %s", key, exc)

    log.info("Cleanup complete — deleted %d object(s)", deleted)
    return 0


if __name__ == "__main__":
    sys.exit(main())
