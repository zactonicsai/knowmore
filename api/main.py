"""FastAPI application entry point."""

from __future__ import annotations
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.services import elasticsearch_svc as es
from api.services import ollama_svc as ollama
from api.routes import documents, search, ai

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown — tolerant of slow service starts."""
    logger.info("Starting up...")

    # Retry ES index creation (ES may still be booting)
    for attempt in range(30):
        try:
            await es.ensure_index()
            logger.info("Elasticsearch index ready")
            break
        except Exception as e:
            logger.warning("ES not ready (attempt %d/30): %s", attempt + 1, e)
            await asyncio.sleep(3)

    # Ollama model pull is best-effort at startup
    try:
        await ollama.ensure_model()
    except Exception as e:
        logger.warning("Ollama model pull deferred: %s", e)

    yield
    logger.info("Shutting down...")
    await es.close()


app = FastAPI(
    title="Document Intelligence Platform",
    description="Upload, process, search, and query documents with AI",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(search.router)
app.include_router(ai.router)


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "document-intelligence-api"}
