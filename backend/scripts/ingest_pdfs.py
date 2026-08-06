"""
Vakalat AI — PDF Ingestion Pipeline
Extracts, cleans, chunks, and embeds all 26 Indian Law Act PDFs into ChromaDB.
"""
import sys
import os
from pathlib import Path
import time
import json
import argparse
import logging

# Ensure UTF-8 stdout encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure backend directory is in python path
CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import (
    ACTS_DIR,
    CHROMA_DIR,
    CHUNKS_DIR,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_PROVIDER,
)
from app.utils.pdf_processor import process_act_file
from app.services.vector_store import VectorStoreService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest_pipeline")


def run_ingestion(reset: bool = False, save_chunks: bool = True, batch_size: int = 100):
    start_time = time.time()

    print("=" * 80)
    print("VAKALAT AI -- DATA INGESTION PIPELINE (PHASE 0)")
    print("=" * 80)
    print(f"Acts Directory:       {ACTS_DIR}")
    print(f"ChromaDB Directory:   {CHROMA_DIR}")
    print(f"Collection Name:      {COLLECTION_NAME}")
    print(f"Embedding Provider:   {EMBEDDING_PROVIDER}")
    print(f"Chunk Size / Overlap: {CHUNK_SIZE} / {CHUNK_OVERLAP}")
    print("=" * 80)

    if not ACTS_DIR.exists():
        logger.error(f"Acts directory not found at: {ACTS_DIR}")
        return False

    pdf_files = sorted([f for f in ACTS_DIR.glob("*.pdf")])
    total_files = len(pdf_files)

    if total_files == 0:
        logger.error(f"No PDF files found in {ACTS_DIR}")
        return False

    print(f"\nFound {total_files} Act PDFs to process.\n")

    # Initialize Vector Store Service
    vector_service = VectorStoreService()
    if reset:
        print("[*] Resetting ChromaDB collection...")
        vector_service.reset_collection()

    all_documents = []
    stats_summary = []

    # Process each PDF
    for idx, pdf_path in enumerate(pdf_files, 1):
        file_start = time.time()
        try:
            docs = process_act_file(pdf_path, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            all_documents.extend(docs)
            file_time = time.time() - file_start

            act_name = docs[0].metadata["act_name"] if docs else pdf_path.stem
            pages_count = docs[-1].metadata.get("page_end", 0) if docs else 0

            stats_summary.append({
                "act_name": act_name,
                "filename": pdf_path.name,
                "chunks": len(docs),
                "pages": pages_count,
            })

            # Save intermediate chunks if requested
            if save_chunks and docs:
                chunk_file = CHUNKS_DIR / f"{pdf_path.stem}.json"
                chunk_data = [
                    {"content": d.page_content, "metadata": d.metadata}
                    for d in docs
                ]
                with open(chunk_file, "w", encoding="utf-8") as cf:
                    json.dump(chunk_data, cf, indent=2, ensure_ascii=False)

            print(f"[{idx:>2}/{total_files}] [OK] {act_name[:45]:<45} | {pages_count:>3} pages | {len(docs):>4} chunks | {file_time:.2f}s")

        except Exception as e:
            print(f"[{idx:>2}/{total_files}] [ERR] Error processing {pdf_path.name}: {e}")

    total_chunks = len(all_documents)
    print("\n" + "-" * 80)
    print(f"Total Chunks Extracted: {total_chunks:,}")
    print("-" * 80)

    # Ingest into ChromaDB
    print(f"\nGenerating embeddings & indexing into ChromaDB (batches of {batch_size})...")
    embed_start = time.time()
    vector_service.add_documents(all_documents, batch_size=batch_size)
    embed_time = time.time() - embed_start

    print(f"[OK] Embeddings indexed in {embed_time:.2f} seconds.")

    # Verify collection stats
    stats = vector_service.get_collection_stats()
    total_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("INGESTION COMPLETE -- SUMMARY REPORT")
    print("=" * 80)
    print(f"Total PDFs Processed:  {total_files}")
    print(f"Total Chunks Created:  {total_chunks:,}")
    print(f"ChromaDB Documents:    {stats.get('document_count', 'N/A')}")
    print(f"Persist Directory:     {stats.get('persist_directory')}")
    print(f"Total Elapsed Time:    {total_time:.2f} seconds ({total_time / 60:.1f} mins)")
    print("=" * 80)

    # Smoke Test
    print("\nRunning Smoke Test Query: 'What is the punishment for theft?'")
    test_results = vector_service.similarity_search_with_score("What is the punishment for theft?", k=3)
    for r_idx, (doc, score) in enumerate(test_results, 1):
        act = doc.metadata.get("act_name", "Unknown")
        pages = doc.metadata.get("pages", "N/A")
        print(f"\n   [Result {r_idx}] Act: {act} (Pages: {pages}) | Distance: {score:.4f}")
        preview = doc.page_content[:200].replace("\n", " ")
        print(f"   Excerpt: \"{preview}...\"")

    print("\nPipeline execution finished successfully!\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Indian Law PDFs into ChromaDB")
    parser.add_argument("--reset", action="store_true", help="Reset ChromaDB collection before ingestion")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for embedding generation")
    parser.add_argument("--no-save-chunks", action="store_true", help="Do not save chunk JSON files")

    args = parser.parse_args()
    run_ingestion(reset=args.reset, save_chunks=not args.no_save_chunks, batch_size=args.batch_size)
