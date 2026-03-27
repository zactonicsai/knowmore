"""LocalStack S3 service for raw file backup."""

from __future__ import annotations
import logging
import boto3
from botocore.config import Config
from api.config import settings

logger = logging.getLogger(__name__)

_client = None


def get_client():
    global _client
    if _client is None:
        _client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint,
            aws_access_key_id="test",
            aws_secret_access_key="test",
            region_name="us-east-1",
            config=Config(signature_version="s3v4"),
        )
        # Ensure bucket exists
        try:
            _client.head_bucket(Bucket=settings.s3_bucket)
        except Exception:
            _client.create_bucket(Bucket=settings.s3_bucket)
            logger.info("Created S3 bucket: %s", settings.s3_bucket)
    return _client


def upload_file(file_bytes: bytes, key: str, content_type: str = "application/octet-stream"):
    client = get_client()
    client.put_object(
        Bucket=settings.s3_bucket,
        Key=key,
        Body=file_bytes,
        ContentType=content_type,
    )
    logger.info("Uploaded to S3: %s", key)


def download_file(key: str) -> bytes:
    client = get_client()
    resp = client.get_object(Bucket=settings.s3_bucket, Key=key)
    return resp["Body"].read()
