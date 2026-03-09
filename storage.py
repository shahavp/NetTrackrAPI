"""
Railway Bucket client — S3-compatible object storage.

Environment variables (set via Railway reference variables):
    BUCKET_NAME              →  ${{tracking-bucket.BUCKET}}
    BUCKET_ENDPOINT          →  ${{tracking-bucket.ENDPOINT}}
    BUCKET_ACCESS_KEY_ID     →  ${{tracking-bucket.ACCESS_KEY_ID}}
    BUCKET_SECRET_ACCESS_KEY →  ${{tracking-bucket.SECRET_ACCESS_KEY}}
    BUCKET_REGION            →  ${{tracking-bucket.REGION}}

Bucket key conventions
----------------------
  uploads/{job_id}.mp4              raw upload (deleted after worker ingests it)
  outputs/{job_id}/trajectory.mp4   trajectory overlay video
  outputs/{job_id}/detection.mp4    detection-annotated video
  outputs/{job_id}/trajectory.csv   raw per-frame CSV
"""

import os

import boto3
from botocore.client import Config

BUCKET = os.environ["BUCKET_NAME"]

_PRESIGN_EXPIRES = int(os.getenv("PRESIGN_EXPIRES_SECONDS", "900"))   # 15 min


def _client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["BUCKET_ENDPOINT"],
        aws_access_key_id=os.environ["BUCKET_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["BUCKET_SECRET_ACCESS_KEY"],
        region_name=os.getenv("BUCKET_REGION", "auto"),
        config=Config(signature_version="s3v4"),
    )


# ---------------------------------------------------------------------------
#  Presigned URLs
# ---------------------------------------------------------------------------

def presigned_upload_url(key: str, expires: int = _PRESIGN_EXPIRES) -> str:
    """Return a presigned PUT URL for direct client → bucket upload."""
    return _client().generate_presigned_url(
        "put_object",
        Params={"Bucket": BUCKET, "Key": key},
        ExpiresIn=expires,
    )


def presigned_download_url(key: str, expires: int = _PRESIGN_EXPIRES) -> str:
    """Return a presigned GET URL for direct client ← bucket download."""
    return _client().generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET, "Key": key},
        ExpiresIn=expires,
    )


# ---------------------------------------------------------------------------
#  Server-side transfers (used by worker and API upload path)
# ---------------------------------------------------------------------------

def upload_file(local_path: str, key: str) -> None:
    _client().upload_file(local_path, BUCKET, key)


def upload_fileobj(fileobj, key: str) -> None:
    _client().upload_fileobj(fileobj, BUCKET, key)


def download_file(key: str, local_path: str) -> None:
    _client().download_file(BUCKET, key, local_path)


def delete_object(key: str) -> None:
    _client().delete_object(Bucket=BUCKET, Key=key)


def list_objects(prefix: str) -> list[dict]:
    """Return all objects under *prefix* (handles pagination)."""
    c = _client()
    paginator = c.get_paginator("list_objects_v2")
    results = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        results.extend(page.get("Contents", []))
    return results


def object_exists(key: str) -> bool:
    try:
        _client().head_object(Bucket=BUCKET, Key=key)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
#  Convenience: job-scoped helpers
# ---------------------------------------------------------------------------

def upload_key(job_id: str) -> str:
    return f"uploads/{job_id}.mp4"


def output_key(job_id: str, filename: str) -> str:
    return f"outputs/{job_id}/{filename}"
