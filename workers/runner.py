"""Temporal worker process entry point."""

from __future__ import annotations
import asyncio
import logging

from temporalio.client import Client
from temporalio.worker import Worker

from workers.workflows import DocumentProcessingWorkflow
from workers.activities import (
    detect_file_type,
    extract_text,
    ocr_extract,
    post_process_text,
    generate_embedding,
    index_to_elasticsearch,
    index_to_chromadb,
)
from api.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def main():
    logger.info("Connecting to Temporal at %s", settings.temporal_host)

    # Retry connection to Temporal
    client = None
    for attempt in range(30):
        try:
            client = await Client.connect(settings.temporal_host)
            logger.info("Connected to Temporal")
            break
        except Exception as e:
            logger.warning("Temporal not ready (attempt %d): %s", attempt + 1, e)
            await asyncio.sleep(2)

    if client is None:
        logger.error("Failed to connect to Temporal after 30 attempts")
        return

    worker = Worker(
        client,
        task_queue="document-processing",
        workflows=[DocumentProcessingWorkflow],
        activities=[
            detect_file_type,
            extract_text,
            ocr_extract,
            post_process_text,
            generate_embedding,
            index_to_elasticsearch,
            index_to_chromadb,
        ],
    )

    logger.info("Starting worker on task queue: document-processing")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
