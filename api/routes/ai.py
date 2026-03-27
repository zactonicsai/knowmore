"""AI assistant endpoints: summarize and question answering."""

from __future__ import annotations
import logging

from fastapi import APIRouter, HTTPException

from api.config import settings
from api.schemas import AIQueryRequest, AIQueryResponse
from api.services import elasticsearch_svc as es
from api.services import chroma_svc as chroma
from api.services import embedding_svc as embed
from api.services import ollama_svc as ollama

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai", tags=["ai"])


def _build_context(es_hits: list[dict], chroma_hits: list[dict], max_chars: int = 8000) -> tuple[str, list[str]]:
    """Merge ES and ChromaDB results into a deduplicated context string."""
    seen_ids = set()
    chunks = []
    sources = []

    for hit in es_hits:
        doc_id = hit["id"]
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            chunks.append(hit.get("text", ""))
            sources.append(hit.get("filename", doc_id))

    for hit in chroma_hits:
        doc_id = hit["id"]
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            chunks.append(hit.get("text", ""))
            sources.append(hit.get("metadata", {}).get("filename", doc_id))

    # Trim to token budget (rough char estimate)
    context = ""
    for chunk in chunks:
        if len(context) + len(chunk) > max_chars:
            remaining = max_chars - len(context)
            if remaining > 100:
                context += chunk[:remaining] + "..."
            break
        context += chunk + "\n\n---\n\n"

    return context.strip(), sources


@router.post("/query", response_model=AIQueryResponse)
async def ai_query(req: AIQueryRequest):
    """AI-powered query: summarize or answer questions from indexed documents."""

    # If summarizing a specific document
    if req.mode == "summarize" and req.document_id:
        doc = await es.get_document(req.document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        context = doc.get("text", "")
        if not context:
            raise HTTPException(status_code=400, detail="Document has no extracted text")
        answer = await ollama.summarize(context[:8000])
        return AIQueryResponse(answer=answer, sources=[doc.get("filename", "")], context_used=1)

    # Retrieve context from both stores
    es_hits = await es.search_keyword(
        query=req.question,
        limit=req.limit,
        category=req.category,
        classification=req.classification,
    )

    query_emb = embed.generate_embedding(req.question)
    where = {}
    if req.category:
        where["category"] = req.category
    chroma_hits = chroma.search_semantic(
        query_embedding=query_emb,
        limit=req.limit,
        where=where if where else None,
    )

    context, sources = _build_context(es_hits, chroma_hits)

    if not context.strip():
        return AIQueryResponse(
            answer="Not found in provided documents. No relevant documents are indexed yet.",
            sources=[],
            context_used=0,
        )

    # Generate response
    if req.mode == "summarize":
        answer = await ollama.summarize(context)
    else:
        answer = await ollama.answer_question(context, req.question)

    return AIQueryResponse(answer=answer, sources=sources, context_used=len(sources))
