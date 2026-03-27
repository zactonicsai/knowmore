"""ChromaDB service for embedding storage and semantic search."""

from __future__ import annotations
import logging
import chromadb
from chromadb.config import Settings as ChromaSettings
from api.config import settings

logger = logging.getLogger(__name__)

_client: chromadb.HttpClient | None = None
_collection = None


def get_client() -> chromadb.HttpClient:
    global _client
    if _client is None:
        host = settings.chromadb_url.replace("http://", "").split(":")[0]
        port = int(settings.chromadb_url.split(":")[-1])
        _client = chromadb.HttpClient(
            host=host,
            port=port,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


def get_collection():
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def add_embedding(doc_id: str, embedding: list[float], text: str, metadata: dict):
    collection = get_collection()
    collection.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[metadata],
    )
    logger.info("Added embedding for %s", doc_id)


def search_semantic(
    query_embedding: list[float],
    limit: int = 10,
    where: dict | None = None,
) -> list[dict]:
    collection = get_collection()
    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": limit,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    output = []
    if results and results["ids"]:
        for i, doc_id in enumerate(results["ids"][0]):
            output.append(
                {
                    "id": doc_id,
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                    "score": 1 - (results["distances"][0][i] if results["distances"] else 0),
                }
            )
    return output
