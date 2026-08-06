"""
Services package for Vakalat AI backend.
"""
from .vector_store import VectorStoreService, get_vector_store_service

__all__ = ["VectorStoreService", "get_vector_store_service"]
