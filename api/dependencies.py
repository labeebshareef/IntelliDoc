from functools import lru_cache
from typing import Optional
import os
from processing import DocumentProcessor, ChatEngine

@lru_cache()
def get_settings():
    """Get application settings."""
    return {
        "STORAGE_PATH": os.getenv("STORAGE_PATH", "./storage"),
        "OLLAMA_HOST": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL", "mistral"),
    }

@lru_cache()
def get_document_processor() -> DocumentProcessor:
    """Get or create DocumentProcessor instance."""
    settings = get_settings()
    return DocumentProcessor(
        storage_path=settings["STORAGE_PATH"]
    )

@lru_cache()
def get_chat_engine() -> ChatEngine:
    """Get or create ChatEngine instance."""
    settings = get_settings()
    return ChatEngine(
        model_name=settings["OLLAMA_MODEL"],
        ollama_base_url=settings["OLLAMA_HOST"]
    )