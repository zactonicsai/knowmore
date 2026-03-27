"""Temporal client for triggering document processing workflows."""

from __future__ import annotations
import logging
from temporalio.client import Client
from api.config import settings

logger = logging.getLogger(__name__)

_client: Client | None = None


async def get_client() -> Client:
    global _client
    if _client is None:
        _client = await Client.connect(settings.temporal_host)
    return _client


async def start_processing_workflow(doc_id: str, filename: str, filepath: str, metadata: dict):
    """Start document processing workflow."""
    client = await get_client()
    handle = await client.start_workflow(
        "DocumentProcessingWorkflow",
        {
            "doc_id": doc_id,
            "filename": filename,
            "filepath": filepath,
            "metadata": metadata,
        },
        id=f"doc-process-{doc_id}",
        task_queue="document-processing",
    )
    logger.info("Started workflow %s for doc %s", handle.id, doc_id)
    return handle.id
