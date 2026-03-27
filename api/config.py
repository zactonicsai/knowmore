"""Application configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    elasticsearch_url: str = "http://localhost:9200"
    chromadb_url: str = "http://localhost:8100"
    temporal_host: str = "localhost:7233"
    ollama_url: str = "http://localhost:11434"
    s3_endpoint: str = "http://localhost:4566"
    s3_bucket: str = "documents"
    upload_dir: str = "/uploads"

    es_index: str = "documents"
    chroma_collection: str = "documents"
    ollama_model: str = "tinyllama"

    max_context_tokens: int = 2048
    max_search_results: int = 5

    class Config:
        env_file = ".env"


settings = Settings()
