"""
Embedding Provider Module for Vakalat AI
Supports Google Gemini Embeddings and Offline Local Sentence-Transformers.
"""
from typing import List
import logging
from langchain_core.embeddings import Embeddings

from ..config import (
    GOOGLE_API_KEY,
    EMBEDDING_PROVIDER,
    GEMINI_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)


class LocalSentenceTransformerEmbeddings(Embeddings):
    """
    Local embedding wrapper using sentence-transformers (offline, fast, free).
    """

    def __init__(self, model_name: str = LOCAL_EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading local embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        if not text:
            return []
        embedding = self.model.encode(
            text,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.tolist()


def get_embedding_function() -> Embeddings:
    """
    Returns the appropriate LangChain-compatible embedding function
    based on configuration and available API keys.
    """
    # 1. Check if Gemini is requested and API key is present
    if (
        EMBEDDING_PROVIDER == "gemini" or (GOOGLE_API_KEY and EMBEDDING_PROVIDER != "local")
    ):
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            logger.info(f"Using Google Gemini embeddings ({GEMINI_EMBEDDING_MODEL})")
            return GoogleGenerativeAIEmbeddings(
                model=GEMINI_EMBEDDING_MODEL,
                google_api_key=GOOGLE_API_KEY,
            )
        except Exception as e:
            logger.warning(
                f"Failed to initialize Gemini embeddings ({e}). Falling back to local sentence-transformers."
            )

    # 2. Default / Fallback: Local SentenceTransformer
    return LocalSentenceTransformerEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)
