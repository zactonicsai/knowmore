"""Temporal activities for document processing pipeline."""

from __future__ import annotations
import logging
import os
import re
import magic
from dataclasses import dataclass

from temporalio import activity

logger = logging.getLogger(__name__)


@dataclass
class ProcessingInput:
    doc_id: str
    filename: str
    filepath: str
    metadata: dict


# ── Activity 1: File Type Detection ───────────────────
@activity.defn
async def detect_file_type(filepath: str) -> dict:
    """Detect MIME type and determine processing strategy."""
    logger.info("Detecting file type: %s", filepath)
    mime = magic.from_file(filepath, mime=True)
    ext = os.path.splitext(filepath)[1].lower()

    needs_ocr = mime in ("image/png", "image/jpeg", "image/tiff", "image/bmp")
    is_pdf = mime == "application/pdf"
    is_text = mime.startswith("text/") or ext in (".txt", ".csv", ".md", ".json")
    is_office = ext in (".docx", ".doc", ".xlsx", ".xls", ".pptx")

    return {
        "mime_type": mime,
        "extension": ext,
        "needs_ocr": needs_ocr,
        "is_pdf": is_pdf,
        "is_text": is_text,
        "is_office": is_office,
    }


# ── Activity 2: Text Extraction (Unstructured) ────────
@activity.defn
async def extract_text(filepath: str) -> str:
    """Extract text from documents using unstructured."""
    logger.info("Extracting text: %s", filepath)
    try:
        from unstructured.partition.auto import partition
        elements = partition(filename=filepath)
        text = "\n\n".join([str(el) for el in elements])
        return text
    except Exception as e:
        logger.warning("Unstructured extraction failed: %s, falling back to raw read", e)
        try:
            with open(filepath, "r", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""


# ── Activity 3: OCR (Tesseract) ───────────────────────
@activity.defn
async def ocr_extract(filepath: str) -> str:
    """OCR extraction for images and scanned PDFs."""
    logger.info("Running OCR: %s", filepath)
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(filepath)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        logger.error("OCR failed: %s", e)
        return ""


# ── Activity 4: Post-Processing ───────────────────────
@activity.defn
async def post_process_text(text: str) -> str:
    """Clean and normalize extracted text."""
    logger.info("Post-processing text (%d chars)", len(text))
    if not text:
        return ""

    # Normalize whitespace
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip control characters (except newlines/tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Trim
    text = text.strip()

    return text


# ── Activity 5: Generate Embeddings ───────────────────
@activity.defn
async def generate_embedding(text: str) -> list[float]:
    """Generate embedding vector for the document text."""
    logger.info("Generating embedding (%d chars)", len(text))
    from api.services.embedding_svc import generate_embedding as gen_emb

    # Use first 512 tokens worth of text for embedding (~2048 chars)
    truncated = text[:2048]
    return gen_emb(truncated)


# ── Activity 6: Index to Elasticsearch ────────────────
@activity.defn
async def index_to_elasticsearch(doc_id: str, text: str, metadata: dict):
    """Store processed text and metadata in Elasticsearch."""
    logger.info("Indexing to ES: %s", doc_id)
    from api.services.elasticsearch_svc import get_client
    from api.config import settings
    from datetime import datetime, timezone

    client = await get_client()
    body = {
        "filename": metadata.get("filename", ""),
        "text": text,
        "category": metadata.get("category", ""),
        "classification": metadata.get("classification", ""),
        "research_area": metadata.get("research_area", ""),
        "status": "indexed",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await client.index(index=settings.es_index, id=doc_id, document=body)


# ── Activity 7: Index to ChromaDB ─────────────────────
@activity.defn
async def index_to_chromadb(doc_id: str, embedding: list[float], text: str, metadata: dict):
    """Store embedding in ChromaDB."""
    logger.info("Indexing to ChromaDB: %s", doc_id)
    from api.services.chroma_svc import add_embedding

    chroma_meta = {
        "filename": metadata.get("filename", ""),
        "category": metadata.get("category", ""),
        "classification": metadata.get("classification", ""),
        "research_area": metadata.get("research_area", ""),
    }
    # ChromaDB document text limited to avoid oversized entries
    add_embedding(doc_id, embedding, text[:10000], chroma_meta)
