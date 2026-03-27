"""Search endpoints: keyword, semantic, hybrid."""

from __future__ import annotations

from fastapi import APIRouter

from api.schemas import SearchRequest, SearchResponse, SearchResult, DocumentMetadata
from api.services import elasticsearch_svc as es
from api.services import chroma_svc as chroma
from api.services import embedding_svc as embed

router = APIRouter(prefix="/api/search", tags=["search"])


@router.post("/", response_model=SearchResponse)
async def search_documents(req: SearchRequest):
    """Unified search endpoint."""
    if req.search_type == "keyword":
        return await _keyword_search(req)
    elif req.search_type == "semantic":
        return await _semantic_search(req)
    else:
        return await _hybrid_search(req)


async def _keyword_search(req: SearchRequest) -> SearchResponse:
    hits = await es.search_keyword(
        query=req.query,
        limit=req.limit,
        category=req.category,
        classification=req.classification,
        research_area=req.research_area,
    )
    results = [
        SearchResult(
            id=h["id"],
            filename=h["filename"],
            text_preview=h["text"][:300] if h["text"] else "",
            score=h["score"],
            metadata=DocumentMetadata(
                category=h.get("category", ""),
                classification=h.get("classification", ""),
                research_area=h.get("research_area", ""),
            ),
        )
        for h in hits
    ]
    return SearchResponse(results=results, total=len(results), search_type="keyword")


async def _semantic_search(req: SearchRequest) -> SearchResponse:
    query_emb = embed.generate_embedding(req.query)

    where = {}
    if req.category:
        where["category"] = req.category
    if req.classification:
        where["classification"] = req.classification

    hits = chroma.search_semantic(
        query_embedding=query_emb,
        limit=req.limit,
        where=where if where else None,
    )
    results = [
        SearchResult(
            id=h["id"],
            filename=h.get("metadata", {}).get("filename", ""),
            text_preview=h["text"][:300] if h["text"] else "",
            score=h["score"],
            metadata=DocumentMetadata(
                category=h.get("metadata", {}).get("category", ""),
                classification=h.get("metadata", {}).get("classification", ""),
                research_area=h.get("metadata", {}).get("research_area", ""),
            ),
        )
        for h in hits
    ]
    return SearchResponse(results=results, total=len(results), search_type="semantic")


async def _hybrid_search(req: SearchRequest) -> SearchResponse:
    """Combine keyword + semantic, deduplicate by doc id, re-rank."""
    kw_resp = await _keyword_search(req)
    sem_resp = await _semantic_search(req)

    seen = {}
    for r in kw_resp.results:
        seen[r.id] = r

    for r in sem_resp.results:
        if r.id in seen:
            # Boost score if found in both
            seen[r.id].score = max(seen[r.id].score, r.score) * 1.2
        else:
            seen[r.id] = r

    combined = sorted(seen.values(), key=lambda x: x.score, reverse=True)[: req.limit]
    return SearchResponse(results=combined, total=len(combined), search_type="hybrid")
