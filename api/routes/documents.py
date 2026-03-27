"""Document upload, listing, and retrieval endpoints."""

from __future__ import annotations
import logging
import os
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import aiofiles

from api.config import settings
from api.schemas import DocumentMetadata, DocumentResponse, UploadResponse
from api.services import elasticsearch_svc as es
from api.services import s3_svc as s3
from api.services import temporal_svc as temporal

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    category: str = Form(""),
    classification: str = Form(""),
    research_area: str = Form(""),
):
    """Upload a document for processing."""
    doc_id = str(uuid.uuid4())
    filename = file.filename or "unknown"

    # Save to upload directory
    os.makedirs(settings.upload_dir, exist_ok=True)
    filepath = os.path.join(settings.upload_dir, f"{doc_id}_{filename}")

    content = await file.read()

    async with aiofiles.open(filepath, "wb") as f:
        await f.write(content)

    # Backup to S3
    try:
        s3.upload_file(content, f"{doc_id}/{filename}", file.content_type or "application/octet-stream")
    except Exception as e:
        # S3 backup is best-effort
        logger.warning("S3 backup failed: %s", e)

    # Index placeholder in ES
    doc_body = {
        "filename": filename,
        "text": "",
        "category": category,
        "classification": classification,
        "research_area": research_area,
        "status": "processing",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await es.index_document(doc_id, doc_body)

    # Start Temporal workflow
    metadata = {"category": category, "classification": classification, "research_area": research_area}
    try:
        await temporal.start_processing_workflow(doc_id, filename, filepath, metadata)
    except Exception as e:
        # If Temporal is down, mark as needing retry
        logger.warning("Temporal workflow start failed: %s", e)
        doc_body["status"] = "pending_retry"
        await es.index_document(doc_id, doc_body)

    return UploadResponse(id=doc_id, filename=filename, status="processing", message="Document uploaded and processing started.")


@router.get("/", response_model=list[DocumentResponse])
async def list_documents():
    """List all indexed documents."""
    docs = await es.list_documents(limit=200)
    results = []
    for doc in docs:
        text = doc.get("text", "")
        results.append(
            DocumentResponse(
                id=doc["id"],
                filename=doc.get("filename", ""),
                metadata=DocumentMetadata(
                    category=doc.get("category", ""),
                    classification=doc.get("classification", ""),
                    research_area=doc.get("research_area", ""),
                ),
                text_preview=text[:300] if text else "",
                full_text=text,
                status=doc.get("status", "unknown"),
                created_at=doc.get("created_at", ""),
            )
        )
    return results


@router.get("/{doc_id}", response_model=DocumentResponse)
async def get_document(doc_id: str):
    """Get a single document by ID."""
    doc = await es.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    text = doc.get("text", "")
    return DocumentResponse(
        id=doc["id"],
        filename=doc.get("filename", ""),
        metadata=DocumentMetadata(
            category=doc.get("category", ""),
            classification=doc.get("classification", ""),
            research_area=doc.get("research_area", ""),
        ),
        text_preview=text[:300] if text else "",
        full_text=text,
        status=doc.get("status", "unknown"),
        created_at=doc.get("created_at", ""),
    )
