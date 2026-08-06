"""
Retrieval Accuracy Test Suite for Vakalat AI (Phase 0)
Validates that ChromaDB similarity search accurately retrieves relevant Indian Law Acts and sections.
"""
import sys
from pathlib import Path
from typing import List, Dict, Any
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

from app.services.vector_store import VectorStoreService
from app.config import CHROMA_DIR, COLLECTION_NAME

logging.basicConfig(level=logging.INFO, format="%(message)s")


TEST_CASES = [
    {
        "id": "TC-01",
        "category": "Criminal Law / Property Offense",
        "query": "What is the punishment for theft and snatching movable property?",
        "expected_act_keywords": ["BNS", "Bharatiya Nyaya Sanhita", "Nyaya"],
    },
    {
        "id": "TC-02",
        "category": "Child Protection / Cyber Law",
        "query": "Punishment for sexual assault on a minor and child sexual abuse material",
        "expected_act_keywords": ["POCSO", "Care And Protection Of Children", "IT Act"],
    },
    {
        "id": "TC-03",
        "category": "Taxation / Indirect Tax",
        "query": "What are the rules and turnover threshold for mandatory GST registration?",
        "expected_act_keywords": ["CGST", "GST"],
    },
    {
        "id": "TC-04",
        "category": "Family / Women's Rights",
        "query": "Protection orders, residence orders, and relief against domestic violence",
        "expected_act_keywords": ["Domestic Violence", "Protection Of Women"],
    },
    {
        "id": "TC-05",
        "category": "Narcotics & Drugs",
        "query": "Punishment for illegal possession, sale, and trafficking of commercial quantity of narcotic drugs",
        "expected_act_keywords": ["NDPS"],
    },
    {
        "id": "TC-06",
        "category": "Corporate Law",
        "query": "Disqualification of directors, duties of board members, and corporate fraud",
        "expected_act_keywords": ["Companies"],
    },
    {
        "id": "TC-07",
        "category": "Civil Law / Agreements",
        "query": "Breach of contract, valid consideration, and compensation for loss or damage",
        "expected_act_keywords": ["Contract"],
    },
]


def run_retrieval_tests(top_k: int = 5) -> bool:
    print("=" * 85)
    print("TEST SUITE: VAKALAT AI -- RETRIEVAL ACCURACY & RELEVANCE")
    print("=" * 85)
    print(f"ChromaDB Path:    {CHROMA_DIR}")
    print(f"Collection:       {COLLECTION_NAME}")
    print(f"Top-K Results:    {top_k}")
    print(f"Total Test Cases: {len(TEST_CASES)}")
    print("=" * 85)

    vector_service = VectorStoreService()
    stats = vector_service.get_collection_stats()
    doc_count = stats.get("document_count", 0)

    if doc_count == 0:
        print("\n[ERR] ChromaDB collection is empty! Please run ingest_pdfs.py first.")
        return False

    print(f"Indexed Chunks in DB: {doc_count:,}\n")

    passed_count = 0

    for tc in TEST_CASES:
        tc_id = tc["id"]
        category = tc["category"]
        query = tc["query"]
        expected_keywords = [kw.lower() for kw in tc["expected_act_keywords"]]

        print(f"\n[{tc_id}] Category: {category}")
        print(f"     Query:    \"{query}\"")
        print(f"     Expected: {', '.join(tc['expected_act_keywords'])}")

        results = vector_service.similarity_search_with_score(query, k=top_k)

        matched = False
        match_rank = -1
        match_act = ""

        print("     Retrieved Top Results:")
        for rank, (doc, score) in enumerate(results, 1):
            act_name = doc.metadata.get("act_name", "Unknown")
            source_file = doc.metadata.get("source_file", "")
            pages = doc.metadata.get("pages", "N/A")
            preview = doc.page_content[:120].replace("\n", " ")

            is_match = any(
                kw in act_name.lower() or kw in source_file.lower()
                for kw in expected_keywords
            )
            marker = "[MATCH]" if is_match else "       "

            if is_match and not matched:
                matched = True
                match_rank = rank
                match_act = act_name

            print(f"       {rank}. {marker} [{act_name}] (Pages {pages}, dist: {score:.4f})")
            print(f"          Snippet: \"{preview}...\"")

        if matched:
            print(f"     Status: [PASS] (Found in Rank #{match_rank} -> {match_act})")
            passed_count += 1
        else:
            print(f"     Status: [FAIL] (Expected act not found in top {top_k} results)")

    pass_rate = (passed_count / len(TEST_CASES)) * 100
    print("\n" + "=" * 85)
    print("TEST SUMMARY REPORT")
    print("=" * 85)
    print(f"Total Tests Run:  {len(TEST_CASES)}")
    print(f"Passed:           {passed_count}")
    print(f"Failed:           {len(TEST_CASES) - passed_count}")
    print(f"Pass Rate:        {pass_rate:.1f}%")
    print(f"Benchmark:        >= 80% required for Phase 0 sign-off")
    print("=" * 85)

    if pass_rate >= 80.0:
        print("\nCHECKPOINT 0.5 PASSED: Retrieval accuracy meets quality standards!\n")
        return True
    else:
        print("\nCHECKPOINT 0.5 FAILED: Retrieval accuracy below 80% threshold.\n")
        return False


if __name__ == "__main__":
    success = run_retrieval_tests(top_k=5)
    sys.exit(0 if success else 1)
