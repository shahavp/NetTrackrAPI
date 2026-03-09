"""
Redis-backed job state.

Replaces the process-local JOBS dict in main.py.  All API replicas and
workers share the same view of job state because it lives in Redis rather
than in one process's heap.

Keys
----
  job:{job_id}  →  JSON-serialised job dict, TTL = JOB_TTL_SECONDS
"""

import json
import os
import time

from redis import Redis

REDIS_URL = os.environ["REDIS_URL"]
JOB_TTL   = int(os.getenv("JOB_TTL_SECONDS", str(30 * 60)))   # 30 min default


def _r() -> Redis:
    return Redis.from_url(REDIS_URL, decode_responses=True)


def _key(job_id: str) -> str:
    return f"job:{job_id}"


# ---------------------------------------------------------------------------
#  CRUD
# ---------------------------------------------------------------------------

def create(job_id: str, data: dict) -> None:
    """Persist a new job dict with full TTL."""
    data = {**data, "_created": time.time()}
    _r().setex(_key(job_id), JOB_TTL, json.dumps(data))


def get(job_id: str) -> dict | None:
    """Return job dict or None if not found / expired."""
    raw = _r().get(_key(job_id))
    return json.loads(raw) if raw else None


def update(job_id: str, **fields) -> None:
    """Merge fields into an existing job, preserving remaining TTL."""
    r   = _r()
    key = _key(job_id)
    raw = r.get(key)
    if not raw:
        return
    data = json.loads(raw)
    data.update(fields)
    ttl = r.ttl(key)
    # Keep at least 60 s so a rapid update just after creation doesn't shrink TTL
    r.setex(key, max(ttl, 60), json.dumps(data))


def delete(job_id: str) -> None:
    _r().delete(_key(job_id))


def exists(job_id: str) -> bool:
    return bool(_r().exists(_key(job_id)))
