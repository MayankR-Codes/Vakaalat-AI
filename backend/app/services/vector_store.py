"""
Vector Store Service for Vakalat AI
Manages persistent ChromaDB vector storage, document indexing, and similarity retrieval.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from ..config import CHROMA_DIR, COLLECTION_NAME
from ..utils.embeddings import get_embedding_function

logger = logging.getLogger(__name__)


class VectorStoreService:
    """
    Service wrapper around ChromaDB vector database.
    """

    def __init__(
        self,
        persist_directory: Path | str = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        embedding_function=None,
    ):
        self.persist_directory = str(persist_directory)
        self.collection_name = collection_name
        self.embedding_function = embedding_function or get_embedding_function()
        self._vector_store: Optional[Chroma] = None

    def get_vector_store(self) -> Chroma:
        """
        Returns or initializes the LangChain Chroma instance.
        """
        if self._vector_store is None:
            self._vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
                persist_directory=self.persist_directory,
            )
        return self._vector_store

    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100,
    ) -> int:
        """
        Adds documents in batches to ChromaDB. Returns count of added documents.
        """
        if not documents:
            return 0

        vs = self.get_vector_store()
        total_docs = len(documents)
        logger.info(f"Adding {total_docs} documents to collection '{self.collection_name}' in batches of {batch_size}...")

        for i in range(0, total_docs, batch_size):
            batch = documents[i : i + batch_size]
            vs.add_documents(batch)

        return total_docs

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Searches for top-k similar documents for a text query.
        """
        vs = self.get_vector_store()
        kwargs = {"k": k}
        if filter_dict:
            kwargs["filter"] = filter_dict
        return vs.similarity_search(query, **kwargs)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Searches for top-k similar documents with distance/similarity scores.
        """
        vs = self.get_vector_store()
        kwargs = {"k": k}
        if filter_dict:
            kwargs["filter"] = filter_dict
        return vs.similarity_search_with_score(query, **kwargs)

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Returns statistics about the Chroma collection.
        """
        vs = self.get_vector_store()
        collection = vs._collection
        count = collection.count() if collection else 0
        return {
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "document_count": count,
        }

    def reset_collection(self) -> None:
        """
        Deletes and recreates the collection.
        """
        vs = self.get_vector_store()
        if vs._collection:
            vs.delete_collection()
            self._vector_store = None
            logger.info(f"Reset collection '{self.collection_name}'.")


def get_vector_store_service() -> VectorStoreService:
    """
    Factory function to get a singleton-ready instance of VectorStoreService.
    """
    return VectorStoreService()
