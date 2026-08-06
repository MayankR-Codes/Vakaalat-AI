"""
PDF Processor for Vakalat AI
Extracts text from Indian Law PDFs, cleans legal formatting, and chunks text with metadata.
"""
from pathlib import Path
from typing import List, Dict, Any
import re
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ..config import CHUNK_SIZE, CHUNK_OVERLAP


def clean_act_name(filename: str) -> str:
    """
    Derives a clean, human-readable Act name from the PDF filename.
    Example: 'protection_of_women_from_domestic_violence_act,_2005.pdf'
             -> 'Protection Of Women From Domestic Violence Act, 2005'
    """
    stem = Path(filename).stem
    # Replace underscores with spaces
    cleaned = stem.replace("_", " ")
    # Clean redundant spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Normalize common acronyms or casing
    acronyms = {"BNS", "BNSS", "BNSA", "POCSO", "NDPS", "IT", "CGST", "SC", "ST", "GST"}
    words = cleaned.split(" ")
    formatted_words = []
    for word in words:
        upper_word = word.upper()
        if upper_word in acronyms or upper_word.rstrip(",") in acronyms:
            formatted_words.append(word.upper())
        elif len(word) > 0 and word.isupper() and len(word) > 4:
            # Title-case long uppercase words like 'PREVENTION' -> 'Prevention'
            formatted_words.append(word.capitalize())
        elif len(word) > 0:
            formatted_words.append(word.capitalize() if not word[0].isupper() else word)
        else:
            formatted_words.append(word)

    return " ".join(formatted_words)


def clean_legal_text(text: str) -> str:
    """
    Cleans raw extracted PDF text while preserving legal structure
    (Sections, Subsections, Chapters, Explanations).
    """
    if not text:
        return ""

    # 1. Replace non-breaking spaces with standard space
    text = text.replace("\xa0", " ")

    # 2. Fix hyphenated words split across lines: 'pun-\nishment' -> 'punishment'
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    # 3. Remove standalone page numbers (e.g. '\n 42 \n')
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)

    # 4. Collapse 3+ consecutive newlines into double newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 5. Collapse excessive spaces or tabs into a single space
    text = re.sub(r"[ \t]{2,}", " ", text)

    # 6. Clean leading/trailing spaces
    return text.strip()


def extract_text_from_pdf(filepath: Path | str) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from a PDF file using PyMuPDF (fitz).
    Returns a list of dicts with 'page_number' and 'text'.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {filepath}")

    doc = fitz.open(str(path))
    pages_data = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        raw_text = page.get_text()
        cleaned_page_text = clean_legal_text(raw_text)
        
        pages_data.append({
            "page_number": page_idx + 1,
            "text": cleaned_page_text,
            "char_count": len(cleaned_page_text),
        })

    doc.close()
    return pages_data


def chunk_document_pages(
    pages_data: List[Dict[str, Any]],
    source_filename: str,
    act_name: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Combines pages and splits into overlapping chunks with metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Combine all page texts with page marker annotations
    full_text = ""
    page_offsets = []  # track char ranges for pages

    for page in pages_data:
        start_char = len(full_text)
        page_text = page["text"]
        if page_text:
            full_text += page_text + "\n\n"
            end_char = len(full_text)
            page_offsets.append({
                "page": page["page_number"],
                "start": start_char,
                "end": end_char,
            })

    if not full_text.strip():
        return []

    # Split text into chunks
    raw_chunks = text_splitter.split_text(full_text)

    documents: List[Document] = []
    current_search_idx = 0

    for idx, chunk_text in enumerate(raw_chunks):
        # Locate approximate pages spanned by this chunk
        chunk_start = full_text.find(chunk_text[:50], current_search_idx)
        if chunk_start == -1:
            chunk_start = current_search_idx
        chunk_end = chunk_start + len(chunk_text)
        current_search_idx = max(0, chunk_start)

        spanned_pages = [
            po["page"]
            for po in page_offsets
            if not (po["end"] < chunk_start or po["start"] > chunk_end)
        ]

        if not spanned_pages:
            spanned_pages = [1]

        metadata = {
            "act_name": act_name,
            "source_file": source_filename,
            "chunk_index": idx,
            "page_start": spanned_pages[0],
            "page_end": spanned_pages[-1],
            "pages": ",".join(map(str, spanned_pages)),
            "char_count": len(chunk_text),
        }

        documents.append(
            Document(page_content=chunk_text, metadata=metadata)
        )

    return documents


def process_act_file(
    filepath: Path | str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    End-to-end processing for a single Act PDF: extraction, cleaning, chunking.
    """
    path = Path(filepath)
    act_name = clean_act_name(path.name)
    pages_data = extract_text_from_pdf(path)
    documents = chunk_document_pages(
        pages_data=pages_data,
        source_filename=path.name,
        act_name=act_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return documents
