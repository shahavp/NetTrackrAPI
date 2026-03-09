"""
RQ job queue — thin wrapper so api.py stays decoupled from RQ internals.

The API calls enqueue_tracking_job().
The worker registers run_tracking_job() in worker.py.

Queue name: "tracking"
"""

import os

from redis import Redis
from rq import Queue

REDIS_URL = os.environ["REDIS_URL"]

# Job result / failure TTL in Redis (seconds) — keep long enough to poll
_RESULT_TTL  = int(os.getenv("JOB_TTL_SECONDS", str(30 * 60)))
_FAILURE_TTL = _RESULT_TTL

_QUEUE_TIMEOUT = int(os.getenv("WORKER_JOB_TIMEOUT", str(30 * 60)))  # max job wall-time


def _queue() -> Queue:
    return Queue("tracking", connection=Redis.from_url(REDIS_URL))


def enqueue_tracking_job(job_id: str, params: dict) -> None:
    """Push a tracking job onto the Redis queue.

    The worker picks it up and calls ``worker.run_tracking_job(job_id, params)``.
    We pass ``job_id`` as the RQ job id so the RQ admin panel shows it correctly.
    """
    _queue().enqueue(
        "worker.run_tracking_job",
        job_id,
        params,
        job_id=job_id,
        result_ttl=_RESULT_TTL,
        failure_ttl=_FAILURE_TTL,
        job_timeout=_QUEUE_TIMEOUT,
    )
