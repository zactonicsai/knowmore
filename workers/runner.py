"""Temporal worker process entry point."""

from __future__ import annotations

import asyncio
import logging
import os

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

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

TEMPORAL_HOST = os.environ.get("TEMPORAL_HOST", "temporal:7233")
TASK_QUEUE = os.environ.get("TEMPORAL_TASK_QUEUE", "document-processing")
TEMPORAL_CONNECT_RETRIES = int(os.environ.get("TEMPORAL_CONNECT_RETRIES", "30"))
TEMPORAL_CONNECT_DELAY_SECONDS = float(os.environ.get("TEMPORAL_CONNECT_DELAY_SECONDS", "2"))


async def main() -> None:
    logger.info("Connecting to Temporal at %s", TEMPORAL_HOST)

    client = None
    for attempt in range(TEMPORAL_CONNECT_RETRIES):
        try:
            client = await Client.connect(TEMPORAL_HOST)
            logger.info("Connected to Temporal")
            break
        except Exception as e:
            logger.warning(
                "Temporal not ready (attempt %d/%d): %s",
                attempt + 1,
                TEMPORAL_CONNECT_RETRIES,
                e,
            )
            await asyncio.sleep(TEMPORAL_CONNECT_DELAY_SECONDS)

    if client is None:
        logger.error(
            "Failed to connect to Temporal after %d attempts",
            TEMPORAL_CONNECT_RETRIES,
        )
        return

    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
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

    logger.info("Starting worker on task queue: %s", TASK_QUEUE)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())