"""
Scan all 26 PDFs — extract text, count characters, pages, estimate chunks and tokens.
This tells us if Gemini free tier is enough or if we need local embeddings.
"""
import fitz
import os

ACTS_DIR = r"data\Acts"
CHUNK_SIZE = 1000       # characters per chunk
CHUNK_OVERLAP = 200     # overlap between chunks

print("=" * 80)
print("VAKALAT AI — DATA ANALYSIS REPORT")
print("=" * 80)

total_chars = 0
total_pages = 0
total_chunks_est = 0
results = []

for filename in sorted(os.listdir(ACTS_DIR)):
    if not filename.endswith(".pdf"):
        continue
    
    filepath = os.path.join(ACTS_DIR, filename)
    
    try:
        doc = fitz.open(filepath)
        pages = len(doc)
        
        # Extract all text
        full_text = ""
        empty_pages = 0
        for page in doc:
            text = page.get_text()
            if len(text.strip()) < 10:
                empty_pages += 1
            full_text += text
        
        doc.close()
        
        chars = len(full_text)
        words = len(full_text.split())
        
        # Estimate chunks: (total_chars - overlap) / (chunk_size - overlap)
        if chars > CHUNK_SIZE:
            chunks_est = max(1, int((chars - CHUNK_OVERLAP) / (CHUNK_SIZE - CHUNK_OVERLAP)))
        else:
            chunks_est = 1
        
        # Rough token estimate (~4 chars per token for English)
        tokens_est = chars // 4
        
        total_chars += chars
        total_pages += pages
        total_chunks_est += chunks_est
        
        results.append({
            "file": filename,
            "pages": pages,
            "empty_pages": empty_pages,
            "chars": chars,
            "words": words,
            "chunks_est": chunks_est,
            "tokens_est": tokens_est
        })
        
        status = "⚠️ HAS EMPTY PAGES" if empty_pages > 2 else "✅"
        print(f"{status} {filename:<55} | {pages:>4} pages | {chars:>8,} chars | ~{chunks_est:>4} chunks | ~{tokens_est:>8,} tokens")
        
    except Exception as e:
        print(f"❌ {filename:<55} | ERROR: {e}")

print("=" * 80)

total_tokens = total_chars // 4
total_words = sum(r["words"] for r in results)

print(f"\n📊 TOTALS:")
print(f"   PDFs processed:     {len(results)}")
print(f"   Total pages:        {total_pages:,}")
print(f"   Total characters:   {total_chars:,}")
print(f"   Total words:        {total_words:,}")
print(f"   Total tokens (est): {total_tokens:,}")
print(f"   Total chunks (est): {total_chunks_est:,}")

print(f"\n" + "=" * 80)
print(f"🔑 GEMINI FREE TIER ANALYSIS")
print(f"=" * 80)

# Gemini embedding: 15 RPM, each request can batch multiple texts
# LangChain sends batches of texts per request
BATCH_SIZE = 20  # texts per API call (LangChain default-ish)
api_calls_needed = (total_chunks_est + BATCH_SIZE - 1) // BATCH_SIZE
time_at_15rpm = api_calls_needed / 15  # minutes

print(f"\n   Chunks to embed:          {total_chunks_est:,}")
print(f"   Tokens to embed:          {total_tokens:,}")
print(f"   API calls needed (~{BATCH_SIZE}/batch): {api_calls_needed}")
print(f"   Time at 15 RPM:           {time_at_15rpm:.1f} minutes")
print(f"   Gemini free limit:        1,000,000 tokens/min")
print(f"   Your total tokens:        {total_tokens:,}")

if total_tokens < 1_000_000:
    print(f"\n   ✅ VERDICT: Your data fits within Gemini free tier!")
    print(f"      Total tokens ({total_tokens:,}) < 1,000,000 limit")
    print(f"      Estimated time: ~{max(time_at_15rpm, 1):.0f}-{max(time_at_15rpm*2, 2):.0f} minutes (with rate limit pauses)")
    print(f"      Cost: $0.00")
else:
    overage = total_tokens - 1_000_000
    print(f"\n   ⚠️ VERDICT: Your data EXCEEDS Gemini free tier per-minute limit!")
    print(f"      Total tokens ({total_tokens:,}) > 1,000,000 limit")
    print(f"      But you can still do it FREE by adding delays between batches.")
    print(f"      The limit is per-MINUTE, not total. Just slow down the requests.")
    print(f"      Estimated time with delays: ~{max(total_tokens / 500_000, 5):.0f}-{max(total_tokens / 300_000, 10):.0f} minutes")
    print(f"      Cost: Still $0.00 (just takes longer)")

print(f"\n" + "=" * 80)
print(f"💡 SENTENCE-TRANSFORMERS (LOCAL) COMPARISON")  
print(f"=" * 80)
print(f"\n   Model: all-MiniLM-L6-v2")
print(f"   Download size: ~80MB (one-time)")
print(f"   Dimensions: 384 (vs Gemini's 768)")
print(f"   Speed on CPU: ~50-100 chunks/second")
print(f"   Estimated time: {total_chunks_est / 75:.0f} seconds ({total_chunks_est / 75 / 60:.1f} minutes)")
print(f"   Cost: $0.00 forever")
print(f"   Quality: ⭐⭐⭐⭐ (good, but Gemini is better)")
print(f"   Internet needed: Only for first download, then fully offline")

print(f"\n" + "=" * 80)
print(f"🎯 RECOMMENDATION")
print(f"=" * 80)
if total_tokens < 2_000_000:
    print(f"\n   → Gemini free tier will work for your {total_chunks_est:,} chunks.")
    print(f"   → Use Gemini if you want better quality embeddings.")
    print(f"   → Use sentence-transformers if you want zero API dependency + faster.")
    print(f"   → Either way, cost = $0.00")
else:
    print(f"\n   → Your data is large. Gemini will work but needs careful rate limiting.")
    print(f"   → Consider sentence-transformers for faster, offline processing.")
