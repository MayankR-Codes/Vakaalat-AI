"""
Utility package for Vakalat AI backend.
"""
from .pdf_processor import (
    extract_text_from_pdf,
    clean_legal_text,
    chunk_document_pages,
    process_act_file,
    clean_act_name,
)
from .embeddings import get_embedding_function

__all__ = [
    "extract_text_from_pdf",
    "clean_legal_text",
    "chunk_document_pages",
    "process_act_file",
    "clean_act_name",
    "get_embedding_function",
]
