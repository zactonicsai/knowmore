"""Temporal workflow for document processing pipeline."""

from __future__ import annotations
import logging
from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from workers.activities import (
        detect_file_type,
        extract_text,
        ocr_extract,
        post_process_text,
        generate_embedding,
        index_to_elasticsearch,
        index_to_chromadb,
    )

logger = logging.getLogger(__name__)


@workflow.defn
class DocumentProcessingWorkflow:
    """Orchestrates the full document processing pipeline."""

    @workflow.run
    async def run(self, input_data: dict) -> dict:
        doc_id = input_data["doc_id"]
        filename = input_data["filename"]
        filepath = input_data["filepath"]
        metadata = {**input_data["metadata"], "filename": filename}

        retry_policy = {
            "start_to_close_timeout": timedelta(minutes=5),
            "retry_policy": workflow.RetryPolicy(
                initial_interval=timedelta(seconds=2),
                backoff_coefficient=2.0,
                maximum_interval=timedelta(minutes=1),
                maximum_attempts=3,
            ),
        }

        # Step 1: Detect file type
        file_info = await workflow.execute_activity(
            detect_file_type, filepath, **retry_policy
        )
        workflow.logger.info("File type detected: %s", file_info)

        # Step 2: Extract text
        text = ""
        if file_info.get("needs_ocr"):
            text = await workflow.execute_activity(
                ocr_extract, filepath, **retry_policy
            )
        else:
            text = await workflow.execute_activity(
                extract_text, filepath, **retry_policy
            )

        # Step 3: If PDF with no text, try OCR
        if file_info.get("is_pdf") and len(text.strip()) < 50:
            ocr_text = await workflow.execute_activity(
                ocr_extract, filepath, **retry_policy
            )
            if len(ocr_text.strip()) > len(text.strip()):
                text = ocr_text

        # Step 4: Post-process
        text = await workflow.execute_activity(
            post_process_text, text, **retry_policy
        )

        if not text.strip():
            workflow.logger.warning("No text extracted for %s", doc_id)
            return {"doc_id": doc_id, "status": "empty", "text_length": 0}

        # Step 5: Generate embeddings
        embedding = await workflow.execute_activity(
            generate_embedding, text, **retry_policy
        )

        # Step 6: Index to Elasticsearch
        await workflow.execute_activity(
            index_to_elasticsearch, doc_id, text, metadata, **retry_policy
        )

        # Step 7: Index to ChromaDB
        await workflow.execute_activity(
            index_to_chromadb, doc_id, embedding, text, metadata, **retry_policy
        )

        return {"doc_id": doc_id, "status": "indexed", "text_length": len(text)}
