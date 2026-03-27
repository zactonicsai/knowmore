"""Elasticsearch service for document indexing and keyword search."""

from __future__ import annotations
import logging
from elasticsearch import AsyncElasticsearch, NotFoundError
from api.config import settings

logger = logging.getLogger(__name__)

_client: AsyncElasticsearch | None = None

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "filename": {"type": "keyword"},
            "text": {"type": "text", "analyzer": "standard"},
            "category": {"type": "keyword"},
            "classification": {"type": "keyword"},
            "research_area": {"type": "keyword"},
            "status": {"type": "keyword"},
            "created_at": {"type": "date"},
        }
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
}


async def get_client() -> AsyncElasticsearch:
    global _client
    if _client is None:
        _client = AsyncElasticsearch(hosts=[settings.elasticsearch_url])
    return _client


async def ensure_index():
    client = await get_client()
    if not await client.indices.exists(index=settings.es_index):
        await client.indices.create(index=settings.es_index, body=INDEX_MAPPING)
        logger.info("Created ES index: %s", settings.es_index)


async def index_document(doc_id: str, body: dict):
    client = await get_client()
    await client.index(index=settings.es_index, id=doc_id, document=body)
    logger.info("Indexed document %s", doc_id)


async def get_document(doc_id: str) -> dict | None:
    client = await get_client()
    try:
        resp = await client.get(index=settings.es_index, id=doc_id)
        return {"id": resp["_id"], **resp["_source"]}
    except NotFoundError:
        return None


async def search_keyword(
    query: str,
    limit: int = 10,
    category: str | None = None,
    classification: str | None = None,
    research_area: str | None = None,
) -> list[dict]:
    client = await get_client()

    must = [{"match": {"text": {"query": query, "fuzziness": "AUTO"}}}]
    filters = []
    if category:
        filters.append({"term": {"category": category}})
    if classification:
        filters.append({"term": {"classification": classification}})
    if research_area:
        filters.append({"term": {"research_area": research_area}})

    body = {
        "query": {"bool": {"must": must, "filter": filters}},
        "size": limit,
        "_source": ["filename", "text", "category", "classification", "research_area"],
    }

    resp = await client.search(index=settings.es_index, body=body)
    results = []
    for hit in resp["hits"]["hits"]:
        src = hit["_source"]
        results.append(
            {
                "id": hit["_id"],
                "filename": src.get("filename", ""),
                "text": src.get("text", ""),
                "score": hit["_score"],
                "category": src.get("category", ""),
                "classification": src.get("classification", ""),
                "research_area": src.get("research_area", ""),
            }
        )
    return results


async def list_documents(limit: int = 100) -> list[dict]:
    client = await get_client()
    body = {"query": {"match_all": {}}, "size": limit, "sort": [{"created_at": "desc"}]}
    resp = await client.search(index=settings.es_index, body=body)
    results = []
    for hit in resp["hits"]["hits"]:
        src = hit["_source"]
        results.append({"id": hit["_id"], **src})
    return results


async def close():
    global _client
    if _client:
        await _client.close()
        _client = None
