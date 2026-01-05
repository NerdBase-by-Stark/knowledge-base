"""Configuration for the Knowledge Base system."""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os


class Settings(BaseSettings):
    # Qdrant
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_grpc_port: int = Field(default=6334)

    # PostgreSQL
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5433)
    postgres_user: str = Field(default="knowledge")
    postgres_password: str = Field(default="knowledge123")
    postgres_db: str = Field(default="knowledge_base")

    # Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="agentmemory123")

    # Embeddings
    embedding_model: str = Field(default="BAAI/bge-m3")
    embedding_dimension: int = Field(default=1024)

    # LLM (Ollama)
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.1:8b")

    # Processing
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=50)

    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    class Config:
        env_prefix = "KB_"
        env_file = ".env"


settings = Settings()


def get_postgres_url(async_mode: bool = False) -> str:
    """Get PostgreSQL connection URL."""
    driver = "postgresql+asyncpg" if async_mode else "postgresql+psycopg2"
    return f"{driver}://{settings.postgres_user}:{settings.postgres_password}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"


def get_qdrant_url() -> str:
    """Get Qdrant connection URL."""
    return f"http://{settings.qdrant_host}:{settings.qdrant_port}"
