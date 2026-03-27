"""Ollama service for context-grounded AI responses."""

from __future__ import annotations
import asyncio
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


async def _wait_for_ollama(max_attempts: int = 30) -> bool:
    """Wait until Ollama API is reachable."""
    for attempt in range(max_attempts):
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{settings.ollama_url}/api/tags")
                if resp.status_code == 200:
                    return True
        except Exception:
            pass
        logger.info("Waiting for Ollama (attempt %d/%d)...", attempt + 1, max_attempts)
        await asyncio.sleep(3)
    return False


async def _model_exists() -> bool:
    """Check if the configured model is already pulled."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{settings.ollama_url}/api/tags")
            models = [m["name"] for m in resp.json().get("models", [])]
            model = settings.ollama_model
            return any(
                m == model or m == f"{model}:latest" or m.startswith(f"{model}:")
                for m in models
            )
    except Exception:
        return False


async def ensure_model():
    """Pull the model if not present. Handles Ollama's streaming pull response."""
    ready = await _wait_for_ollama()
    if not ready:
        logger.error("Ollama not reachable at %s", settings.ollama_url)
        return

    if await _model_exists():
        logger.info("Ollama model '%s' already available", settings.ollama_model)
        return

    logger.info("Pulling Ollama model '%s' (this may take a few minutes)...", settings.ollama_model)
    try:
        async with httpx.AsyncClient() as client:
            # Ollama /api/pull streams JSON lines — must use streaming request
            async with client.stream(
                "POST",
                f"{settings.ollama_url}/api/pull",
                json={"name": settings.ollama_model},
                timeout=httpx.Timeout(connect=10, read=600, write=10, pool=10),
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.strip():
                        logger.debug("Pull progress: %s", line.strip()[:100])

        if await _model_exists():
            logger.info("Model '%s' pulled successfully", settings.ollama_model)
        else:
            logger.error("Model '%s' pull completed but model not found in tags", settings.ollama_model)
    except Exception as e:
        logger.error("Failed to pull Ollama model: %s", e)


async def generate(prompt: str, system: str = SYSTEM_PROMPT) -> str:
    """Send prompt to Ollama and return response. Retries once on failure."""
    last_error = None

    for attempt in range(2):
        try:
            async with httpx.AsyncClient() as client:
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
                    timeout=httpx.Timeout(connect=10, read=120, write=10, pool=10),
                )
                resp.raise_for_status()
                result = resp.json().get("response", "").strip()
                if result:
                    return result
                return "The model returned an empty response. Please try again."
        except httpx.ConnectError as e:
            last_error = f"Cannot connect to Ollama at {settings.ollama_url}: {e}"
            logger.warning("Ollama connection failed (attempt %d): %s", attempt + 1, e)
        except httpx.HTTPStatusError as e:
            last_error = f"Ollama returned error {e.response.status_code}: {e.response.text[:200]}"
            logger.warning("Ollama HTTP error (attempt %d): %s", attempt + 1, last_error)
        except httpx.TimeoutException as e:
            last_error = f"Ollama request timed out: {e}"
            logger.warning("Ollama timeout (attempt %d): %s", attempt + 1, e)
        except Exception as e:
            last_error = f"Unexpected Ollama error: {e}"
            logger.warning("Ollama error (attempt %d): %s", attempt + 1, e)

        if attempt == 0:
            await asyncio.sleep(2)

    return f"AI service error: {last_error}"


async def summarize(context: str) -> str:
    prompt = SUMMARIZE_PROMPT.format(context=context)
    return await generate(prompt)


async def answer_question(context: str, question: str) -> str:
    prompt = QUESTION_PROMPT.format(context=context, question=question)
    return await generate(prompt)
