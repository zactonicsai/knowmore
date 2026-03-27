# DocIntel — Document Intelligence Platform

A modular system for document ingestion, processing, search, and context-aware AI — built with FastAPI, Temporal, Elasticsearch, ChromaDB, and Ollama.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│   Frontend   │────▶│   FastAPI    │────▶│   Temporal    │
│  (Tailwind)  │     │   Backend    │     │   Workers     │
└─────────────┘     └──────┬──────┘     └──────┬───────┘
                           │                    │
              ┌────────────┼────────────┐       │
              ▼            ▼            ▼       ▼
        ┌──────────┐ ┌──────────┐ ┌─────────┐ ┌───────┐
        │Elastic-  │ │ ChromaDB │ │  Ollama  │ │  S3   │
        │search    │ │(vectors) │ │ (AI/LLM) │ │backup │
        └──────────┘ └──────────┘ └─────────┘ └───────┘
```

## Services

| Service        | Port  | Purpose                        |
|----------------|-------|--------------------------------|
| Frontend       | 3000  | Web UI (nginx + Tailwind)      |
| API            | 8000  | FastAPI backend                |
| Elasticsearch  | 9200  | Full-text search + metadata    |
| ChromaDB       | 8100  | Vector embeddings              |
| Temporal       | 7233  | Workflow orchestration          |
| Temporal UI    | 8080  | Workflow monitoring             |
| Ollama         | 11434 | Local LLM (tinyllama)          |
| LocalStack     | 4566  | S3-compatible file backup      |

## Quick Start

```bash
# Clone and start
cd project
chmod +x scripts/setup.sh
./scripts/setup.sh
```

Or manually:

```bash
docker compose up -d --build

# Pull the AI model
docker compose exec ollama ollama pull tinyllama
```

Then open **http://localhost:3000**

## Processing Pipeline

Documents flow through Temporal workflow activities:

1. **File Type Detection** — MIME type identification
2. **Text Extraction** — via Unstructured library
3. **OCR** — Tesseract fallback for images/scanned PDFs
4. **Post-Processing** — whitespace normalization, cleanup
5. **Embedding Generation** — sentence-transformers (all-MiniLM-L6-v2)
6. **Elasticsearch Indexing** — full text + metadata
7. **ChromaDB Indexing** — vector embeddings

## Search Types

- **Keyword** — Elasticsearch full-text with fuzzy matching
- **Semantic** — ChromaDB cosine similarity on embeddings
- **Hybrid** — Combined keyword + semantic with deduplication and score boosting

## AI Features

- **Question Answering** — context-grounded responses from indexed documents
- **Summarization** — document or corpus summaries
- **Strict grounding** — AI only uses provided context, never external knowledge

## API Endpoints

| Method | Endpoint              | Description                |
|--------|-----------------------|----------------------------|
| POST   | /api/documents/upload | Upload document            |
| GET    | /api/documents/       | List all documents         |
| GET    | /api/documents/{id}   | Get single document        |
| POST   | /api/search/          | Search (keyword/semantic)  |
| POST   | /ai/query             | AI question/summarize      |
| GET    | /api/health           | Health check               |

## Testing

```bash
# Run tests (inside API container)
docker compose exec api pytest tests/ -v

# Or locally
pip install -r requirements.txt
pytest tests/ -v
```

## Domain Focus

Optimized for grocery and food data analysis:
- Price tracking (fruits, vegetables, proteins, staples)
- Trend identification
- Meal planning from available data
- Report summarization

## Configuration

Environment variables (set in docker-compose.yml):

| Variable          | Default                     |
|-------------------|-----------------------------|
| ELASTICSEARCH_URL | http://elasticsearch:9200   |
| CHROMADB_URL      | http://chromadb:8100        |
| TEMPORAL_HOST     | temporal:7233               |
| OLLAMA_URL        | http://ollama:11434         |
| S3_ENDPOINT       | http://localstack:4566      |
| OLLAMA_MODEL      | tinyllama                   |
"# knowmore" 
