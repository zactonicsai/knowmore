"""Request/response schemas."""

from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ── Upload ─────────────────────────────────────────────
class DocumentMetadata(BaseModel):
    category: str = ""
    classification: str = ""
    research_area: str = ""


class DocumentResponse(BaseModel):
    id: str
    filename: str
    metadata: DocumentMetadata
    text_preview: str = ""
    full_text: str = ""
    status: str = "pending"
    created_at: str = ""


class UploadResponse(BaseModel):
    id: str
    filename: str
    status: str
    message: str


# ── Search ─────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    search_type: str = "keyword"  # keyword | semantic | hybrid
    category: Optional[str] = None
    classification: Optional[str] = None
    research_area: Optional[str] = None
    limit: int = Field(default=10, le=50)


class SearchResult(BaseModel):
    id: str
    filename: str
    text_preview: str
    score: float
    metadata: DocumentMetadata


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total: int
    search_type: str


# ── AI ─────────────────────────────────────────────────
class AIQueryRequest(BaseModel):
    question: str
    mode: str = "question"  # question | summarize
    document_id: Optional[str] = None
    category: Optional[str] = None
    classification: Optional[str] = None
    limit: int = Field(default=5, le=10)


class AIQueryResponse(BaseModel):
    answer: str
    sources: list[str]
    context_used: int
