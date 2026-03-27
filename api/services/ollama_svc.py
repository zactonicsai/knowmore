"""Ollama service for context-grounded AI responses."""

from __future__ import annotations
import logging
import httpx
from api.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful document assistant. You answer questions ONLY using the provided context.

RULES:
1. Answer ONLY using the provided context below.
2. If the answer is not found in the context, respond exactly: "Not found in provided documents."
3. Do not use any external knowledge.
4. Be concise and factual.
5. When summarizing, cover all key points from the context.
6. Cite specific details from the documents when possible."""

SUMMARIZE_PROMPT = """Summarize the following document content. Include all key points, data, and findings.
Only use information from the provided context.

CONTEXT:
{context}

SUMMARY:"""

QUESTION_PROMPT = """Answer the following question using ONLY the provided context.
If the answer is not in the context, say: "Not found in provided documents."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


async def ensure_model():
    """Pull the model if not present."""
    async with httpx.AsyncClient(timeout=600) as client:
        try:
            resp = await client.get(f"{settings.ollama_url}/api/tags")
            models = [m["name"] for m in resp.json().get("models", [])]
            if settings.ollama_model not in models and f"{settings.ollama_model}:latest" not in models:
                logger.info("Pulling model %s ...", settings.ollama_model)
                await client.post(
                    f"{settings.ollama_url}/api/pull",
                    json={"name": settings.ollama_model},
                    timeout=600,
                )
                logger.info("Model %s pulled successfully", settings.ollama_model)
        except Exception as e:
            logger.warning("Could not check/pull Ollama model: %s", e)


async def generate(prompt: str, system: str = SYSTEM_PROMPT) -> str:
    """Send prompt to Ollama and return response."""
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{settings.ollama_url}/api/generate",
            json={
                "model": settings.ollama_model,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 1024,
                },
            },
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()


async def summarize(context: str) -> str:
    prompt = SUMMARIZE_PROMPT.format(context=context)
    return await generate(prompt)


async def answer_question(context: str, question: str) -> str:
    prompt = QUESTION_PROMPT.format(context=context, question=question)
    return await generate(prompt)
