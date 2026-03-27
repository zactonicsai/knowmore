"""Tests for API endpoints and service logic."""

from __future__ import annotations
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from api.main import app
from api.services.ollama_svc import SUMMARIZE_PROMPT, QUESTION_PROMPT


client = TestClient(app)


# ── Health ─────────────────────────────────────────────
class TestHealth:
    def test_health_endpoint(self):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


# ── Documents ─────────────────────────────────────────
class TestDocuments:
    @patch("api.routes.documents.es")
    def test_list_documents_empty(self, mock_es):
        mock_es.list_documents = AsyncMock(return_value=[])
        resp = client.get("/api/documents/")
        assert resp.status_code == 200
        assert resp.json() == []

    @patch("api.routes.documents.es")
    def test_list_documents(self, mock_es):
        mock_es.list_documents = AsyncMock(return_value=[
            {
                "id": "abc-123",
                "filename": "test.pdf",
                "text": "Sample text content here",
                "category": "grocery",
                "classification": "public",
                "research_area": "pricing",
                "status": "indexed",
                "created_at": "2025-01-01T00:00:00Z",
            }
        ])
        resp = client.get("/api/documents/")
        assert resp.status_code == 200
        docs = resp.json()
        assert len(docs) == 1
        assert docs[0]["filename"] == "test.pdf"
        assert docs[0]["metadata"]["category"] == "grocery"

    @patch("api.routes.documents.es")
    def test_get_document_not_found(self, mock_es):
        mock_es.get_document = AsyncMock(return_value=None)
        resp = client.get("/api/documents/nonexistent")
        assert resp.status_code == 404


# ── Search ─────────────────────────────────────────────
class TestSearch:
    @patch("api.routes.search.es")
    def test_keyword_search(self, mock_es):
        mock_es.search_keyword = AsyncMock(return_value=[
            {
                "id": "doc-1",
                "filename": "prices.csv",
                "text": "Chicken price data for Q3",
                "score": 5.2,
                "category": "protein",
                "classification": "",
                "research_area": "pricing",
            }
        ])
        resp = client.post("/api/search/", json={
            "query": "chicken prices",
            "search_type": "keyword",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["search_type"] == "keyword"
        assert len(data["results"]) == 1
        assert data["results"][0]["filename"] == "prices.csv"

    @patch("api.routes.search.embed")
    @patch("api.routes.search.chroma")
    def test_semantic_search(self, mock_chroma, mock_embed):
        mock_embed.generate_embedding = MagicMock(return_value=[0.1] * 384)
        mock_chroma.search_semantic = MagicMock(return_value=[
            {
                "id": "doc-2",
                "text": "Salmon nutrition info",
                "metadata": {"filename": "nutrition.pdf", "category": "protein", "classification": "", "research_area": ""},
                "distance": 0.15,
                "score": 0.85,
            }
        ])
        resp = client.post("/api/search/", json={
            "query": "fish nutrition",
            "search_type": "semantic",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["search_type"] == "semantic"
        assert len(data["results"]) == 1


# ── AI Endpoints ──────────────────────────────────────
class TestAI:
    @patch("api.routes.ai.ollama")
    @patch("api.routes.ai.embed")
    @patch("api.routes.ai.chroma")
    @patch("api.routes.ai.es")
    def test_ai_question(self, mock_es, mock_chroma, mock_embed, mock_ollama):
        mock_es.search_keyword = AsyncMock(return_value=[
            {"id": "d1", "filename": "grocery.txt", "text": "Rice is $2.50 per pound."}
        ])
        mock_embed.generate_embedding = MagicMock(return_value=[0.1] * 384)
        mock_chroma.search_semantic = MagicMock(return_value=[])
        mock_ollama.answer_question = AsyncMock(return_value="Rice costs $2.50 per pound.")

        resp = client.post("/ai/query", json={
            "question": "How much does rice cost?",
            "mode": "question",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "2.50" in data["answer"]
        assert len(data["sources"]) > 0

    @patch("api.routes.ai.ollama")
    @patch("api.routes.ai.es")
    def test_ai_summarize_document(self, mock_es, mock_ollama):
        mock_es.get_document = AsyncMock(return_value={
            "id": "d1",
            "filename": "report.pdf",
            "text": "This report covers grocery pricing trends for Q3 2024.",
        })
        mock_ollama.summarize = AsyncMock(return_value="The report covers Q3 2024 grocery pricing trends.")

        resp = client.post("/ai/query", json={
            "question": "summarize",
            "mode": "summarize",
            "document_id": "d1",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "Q3" in data["answer"]

    @patch("api.routes.ai.embed")
    @patch("api.routes.ai.chroma")
    @patch("api.routes.ai.es")
    def test_ai_no_context(self, mock_es, mock_chroma, mock_embed):
        mock_es.search_keyword = AsyncMock(return_value=[])
        mock_embed.generate_embedding = MagicMock(return_value=[0.1] * 384)
        mock_chroma.search_semantic = MagicMock(return_value=[])

        resp = client.post("/ai/query", json={
            "question": "What is the meaning of life?",
            "mode": "question",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "not found" in data["answer"].lower() or "no relevant" in data["answer"].lower()


# ── Prompt Template Tests ─────────────────────────────
class TestPrompts:
    def test_summarize_prompt_has_context_placeholder(self):
        assert "{context}" in SUMMARIZE_PROMPT

    def test_question_prompt_has_placeholders(self):
        assert "{context}" in QUESTION_PROMPT
        assert "{question}" in QUESTION_PROMPT

    def test_question_prompt_enforces_grounding(self):
        assert "ONLY" in QUESTION_PROMPT
        assert "Not found in provided documents" in QUESTION_PROMPT


# ── Worker Activities ─────────────────────────────────
class TestActivities:
    @pytest.mark.asyncio
    async def test_post_process_text(self):
        from workers.activities import post_process_text
        result = await post_process_text("  Hello   world  \n\n\n\n  test  ")
        assert "Hello world" in result
        assert "\n\n\n\n" not in result

    @pytest.mark.asyncio
    async def test_post_process_empty(self):
        from workers.activities import post_process_text
        result = await post_process_text("")
        assert result == ""
