<div align="center">

# 🏛️ Vakalat AI

### *Legal Intelligence for Every Citizen*

**A production-ready Python backend that identifies Indian law offenses from text or voice descriptions, built on a Retrieval-Augmented Generation (RAG) pipeline over 26 Indian Acts.**

<br/>

[![Python](https://img.shields.io/badge/Python-3.10.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.x-1C3C3C?style=for-the-badge)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.6.x-FF6B35?style=for-the-badge)](https://trychroma.com)
[![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)

<br/>

> ⚠️ **Legal Disclaimer:** Vakalat AI is strictly for **educational and informational purposes only**.
> It does not constitute legal advice. Always consult a qualified legal professional for actual legal matters.

</div>

---

## 📖 Table of Contents

1. [What is Vakalat AI?](#-what-is-vakalat-ai)
2. [How It Works](#-how-it-works)
3. [System Architecture](#-system-architecture)
4. [Indian Acts Coverage](#-indian-acts-coverage)
5. [Tech Stack](#-tech-stack)
6. [Project Structure](#-project-structure)
7. [Getting Started](#-getting-started)
8. [Environment Variables](#-environment-variables)
9. [Running the Data Pipeline](#-running-the-data-pipeline)
10. [Running the Backend Server](#-running-the-backend-server)
11. [API Reference](#-api-reference)
12. [Running Tests](#-running-tests)
13. [Development Roadmap](#-development-roadmap)
14. [Contributing](#-contributing)

---

## 🎯 What is Vakalat AI?

Vakalat AI is a **legal AI backend** designed for the Indian legal system. This repository contains the **Python backend only** — the Flutter mobile app frontend lives in a separate repository.

### The Problem

Most Indian citizens do not know which law applies when a crime is committed against them, which sections are relevant, or what the punishment is. Access to this knowledge typically requires a lawyer.

### The Solution

Vakalat AI allows any citizen to describe their situation in plain language (text or voice, in English or Hindi) and instantly receive a structured legal analysis — identifying the offense, the governing Act, specific sections, and statutory punishment — all grounded in actual Indian law text.

---

### Version Breakdown

| Version | Product | Who It's For | Access |
|---------|---------|--------------|--------|
| **Version 1** | 🔍 **Offense Identifier AI** | General public | Free |
| **Version 2** | 📚 **Law Tutor & Grader AI** | Law students | Subscription |

**Version 1** — A user describes a real-world situation (e.g., *"Someone hacked into my bank account and transferred money"*). The AI retrieves relevant legal text from its vector database, identifies the applicable offense (e.g., IT Act Section 66C), explains it in plain language, and responds in text and voice.

**Version 2** — A law student asks the AI to explain a legal section. The AI generates 5 quiz questions, the student responds (via text, voice, or a photo of handwritten answers), and the AI grades each answer with detailed feedback.

---

## ⚙️ How It Works

Vakalat AI uses a **Retrieval-Augmented Generation (RAG)** pipeline:

```
User Input (text or voice)
        │
        ▼
[1] Speech-to-Text (Whisper)          ← only for voice input
        │
        ▼
[2] Query the Vector Database
    ChromaDB searches 12,239+ legal
    chunks across 26 Indian Acts
    and returns the top-5 most
    relevant passages
        │
        ▼
[3] Gemini 2.5 Flash
    Receives the user query +
    retrieved legal context and
    generates a structured response
    (offense name, sections, punishment,
    plain explanation, next steps)
        │
        ▼
[4] Text-to-Speech (Edge-TTS)         ← converts response to audio
        │
        ▼
JSON Response + MP3 Audio
```

This approach ensures that every response is **grounded in actual Indian law text**, not hallucinated by the language model.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Flutter Mobile App (Separate Repo)                    │
│                                                                         │
│    Text Input Box              Voice Recorder Button                    │
│    (user types situation)      (user speaks in English/Hindi)           │
└──────────────────┬─────────────────────────┬───────────────────────────┘
                   │                         │
                   │ POST /offense/identify  │ POST /audio/transcribe
                   │ application/json        │ multipart/form-data
                   ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     FASTAPI BACKEND (Python 3.10)                       │
│                                                                         │
│   ┌─────────────────────────┐    ┌──────────────────────────────────┐   │
│   │   API Routers + CORS    │    │      Whisper STT Engine          │   │
│   │   (Pydantic v2 schemas) │    │   (audio → transcribed text)     │   │
│   └────────────┬────────────┘    └─────────────────┬────────────────┘   │
│                │                                   │                    │
│                └──────────────────┬────────────────┘                    │
│                                   ▼                                     │
│                    ┌──────────────────────────┐                         │
│                    │  LangChain RAG Controller │                         │
│                    └──────────────┬───────────┘                         │
│                                   │                                     │
│              ┌────────────────────┴──────────────────────┐              │
│              ▼                                           ▼              │
│   ┌──────────────────────┐                  ┌───────────────────────┐   │
│   │  ChromaDB Retriever  │                  │ Google Gemini 2.5 Flash│  │
│   │  12,239+ chunks from │ ───── context ──▶│  Legal Classification │   │
│   │  26 Indian Acts      │                  │  & Response Generator │   │
│   └──────────────────────┘                  └──────────┬────────────┘   │
│                                                        │                │
│                                            ┌───────────▼────────────┐   │
│                                            │   Edge-TTS Engine      │   │
│                                            │   (text → MP3 audio)   │   │
│                                            └───────────┬────────────┘   │
└────────────────────────────────────────────────────────┼────────────────┘
                                                         │
                    ┌────────────────────────────────────┴────────────────┐
                    ▼                                                      ▼
        Structured JSON Response                           Streamable MP3 Audio
        (offense, sections, punishment,                    (Indian English / Hindi
         explanation, recommended action)                   voice response)
```

---

## 📚 Indian Acts Coverage

The RAG vector database is built over **26 Indian Law Acts**, covering criminal, civil, corporate, taxation, cyber, and environmental law:

| # | Act | Year | Domain |
|---|-----|------|--------|
| 1 | Bharatiya Nyaya Sanhita (BNS) | 2023 | Criminal Law (replaces IPC) |
| 2 | Bharatiya Nagarik Suraksha Sanhita (BNSS) | 2023 | Criminal Procedure (replaces CrPC) |
| 3 | Bharatiya Sakshya Adhiniyam (BNSA) | 2023 | Evidence (replaces Indian Evidence Act) |
| 4 | Constitution of India | — | Constitutional Law |
| 5 | Information Technology (IT) Act | 2000 | Cyber Law |
| 6 | POCSO Act | 2012 | Child Protection |
| 7 | NDPS Act | 1985 | Narcotics & Drugs |
| 8 | Protection of Women from Domestic Violence Act | 2005 | Family / Women's Rights |
| 9 | Dowry Prohibition Act | 1961 | Family Law |
| 10 | Companies Act | 2013 | Corporate Law |
| 11 | Indian Contract Act | 1872 | Civil / Contract Law |
| 12 | Prevention of Corruption Act | 1988 | Anti-Corruption |
| 13 | Prevention of Money Laundering Act | 2002 | Financial Crime |
| 14 | Income Tax Act | 1961 | Taxation |
| 15 | CGST Act | 2017 | GST / Indirect Tax |
| 16 | Consumer Protection Act | 2019 | Consumer Rights |
| 17 | Motor Vehicle Act | 1988 | Traffic & Road Safety |
| 18 | Arms Act | 1959 | Weapons |
| 19 | Wildlife Protection Act | 1972 | Environmental Law |
| 20 | Environment (Protection) Act | 1986 | Environmental Law |
| 21 | Copyright Act | 1957 | Intellectual Property |
| 22 | Trade Marks Act | 1999 | Intellectual Property |
| 23 | SC & ST (Prevention of Atrocities) Act | 1989 | Social Justice |
| 24 | Care and Protection of Children Act | — | Child Welfare |
| 25 | Digital Personal Data Protection Act | 2023 | Data Privacy |
| 26 | Disaster Management Act | 2005 | Emergency Law |

---

## 📦 Tech Stack

### Phase 0 — Data Pipeline

| Library | Version | Purpose |
|---------|---------|---------|
| **pymupdf (fitz)** | `1.25.3` | Extracts and cleans text from Indian Law PDFs |
| **langchain** | `0.3.25` | Core LangChain framework for building AI chains |
| **langchain-community** | `0.3.24` | ChromaDB integration with LangChain |
| **sentence-transformers** | `3.3.1` | Local embedding model (`all-MiniLM-L6-v2`) for offline use |
| **chromadb** | `0.6.3` | Persistent vector database for storing 12,239+ legal chunks |
| **google-generativeai** | `0.8.5` | Google Gemini SDK (used for Gemini embeddings option) |
| **python-dotenv** | `1.1.0` | Loads `.env` configuration files |

### Phase 1 — Offense Identifier AI

| Library | Version | Purpose |
|---------|---------|---------|
| **fastapi** | `0.115.0` | High-performance REST API framework |
| **uvicorn[standard]** | `0.34.0` | ASGI server to run FastAPI |
| **python-multipart** | `0.0.20` | Enables audio/file uploads via multipart form data |
| **langchain-google-genai** | `2.1.0` | LangChain integration with Google Gemini |
| **langchain-core** | `0.3.0` | LangChain core abstractions (prompts, chains, parsers) |
| **openai-whisper** | `20240930` | OpenAI's open-source local speech-to-text model |
| **edge-tts** | `7.0.0` | Microsoft Edge TTS — natural Hindi & Indian English voices |

> **System Dependency:** `ffmpeg` must be installed separately for Whisper to decode mobile audio formats (`.m4a`, `.aac`).

---

## 🗂️ Project Structure

```
Vakalat AI/                          ← Project Root
│
├── backend/                         ← FastAPI Python Application
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                  ← FastAPI app + CORS middleware       [Phase 1]
│   │   ├── config.py                ← Centralised config (paths, keys, settings)
│   │   │
│   │   ├── models/                  ← Pydantic v2 data schemas            [Phase 1]
│   │   │   ├── __init__.py
│   │   │   └── offense.py           ← Request & response models
│   │   │
│   │   ├── prompts/                 ← LLM system prompts                  [Phase 1]
│   │   │   ├── __init__.py
│   │   │   └── offense_prompts.py   ← Indian legal expert system prompt
│   │   │
│   │   ├── chains/                  ← LangChain AI chains                 [Phase 1]
│   │   │   ├── __init__.py
│   │   │   └── offense_chain.py     ← RAG chain: query → retrieve → Gemini
│   │   │
│   │   ├── services/                ← Business logic services
│   │   │   ├── __init__.py
│   │   │   ├── vector_store.py      ← ChromaDB: add, search, stats, reset ✅
│   │   │   ├── stt_service.py       ← Whisper speech-to-text              [Phase 1]
│   │   │   └── tts_service.py       ← Edge-TTS text-to-speech             [Phase 1]
│   │   │
│   │   ├── api/                     ← API route handlers                  [Phase 1]
│   │   │   └── v1/
│   │   │       ├── __init__.py
│   │   │       ├── router.py        ← Central router (mounts all sub-routers)
│   │   │       ├── offense.py       ← POST /offense/identify endpoints
│   │   │       └── audio.py         ← POST /audio/transcribe & /synthesize
│   │   │
│   │   └── utils/                   ← Reusable utility modules
│   │       ├── __init__.py
│   │       ├── pdf_processor.py     ← PDF → clean text → LangChain chunks  ✅
│   │       └── embeddings.py        ← Gemini / local embedding selector    ✅
│   │
│   ├── scripts/
│   │   └── ingest_pdfs.py           ← One-time PDF ingestion pipeline      ✅
│   │
│   └── tests/
│       ├── test_retrieval.py        ← ChromaDB retrieval accuracy tests    ✅
│       ├── test_offense_chain.py    ← 20+ offense ID scenario tests        [Phase 1]
│       └── test_audio_pipeline.py   ← STT + TTS integration tests          [Phase 1]
│
├── data/
│   ├── Acts/                        ← 26 Indian Law PDFs                   ✅
│   ├── chroma_db/                   ← ChromaDB persistent vector store     ✅
│   └── chunks/                      ← Processed chunk JSON files           ✅
│
├── check_data.py                    ← PDF analysis & Gemini cost estimator ✅
├── requirements.txt                 ← All Python dependencies
├── .env                             ← Local environment variables (git-ignored)
├── .env.example                     ← Environment variable template
├── .gitignore
├── ROADMAP.md                       ← Full sprint-by-sprint project roadmap
└── PHASE_1_GUIDE.md                 ← Detailed Phase 1 implementation guide
```

> **Legend:** ✅ Built & working &nbsp;|&nbsp; `[Phase 1]` To be built next

---

## 🚀 Getting Started

### Prerequisites

Before you begin, make sure the following are installed on your system:

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Python** | `3.10.x` | Strictly 3.10 — required for Whisper + PyTorch compatibility |
| **Git** | Any | For cloning the repository |
| **ffmpeg** | Any | Required by Whisper to decode mobile audio formats |

#### Install ffmpeg (Windows)
```bash
winget install ffmpeg

# Verify the installation
ffmpeg -version
```

#### Install ffmpeg (macOS)
```bash
brew install ffmpeg
```

---

### Step 1 — Clone the Repository

```bash
git clone <your-repo-url>
cd "Vakalat AI"
```

---

### Step 2 — Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment:

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Installing `openai-whisper` will also download PyTorch (~500MB) on first run. This is a one-time download. `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~80MB) on first use.

---

## 🔑 Environment Variables

Copy the template and configure your values:

```bash
cp .env.example .env
```

Open `.env` and fill in the following:

```env
# ──────────────────────────────────────────────────────────────
# REQUIRED
# ──────────────────────────────────────────────────────────────

# Your Google Gemini API key
# Get one free at: https://aistudio.google.com/apikey
GOOGLE_API_KEY=your_gemini_api_key_here

# ──────────────────────────────────────────────────────────────
# OPTIONAL (defaults shown below)
# ──────────────────────────────────────────────────────────────

# Embedding provider: "gemini" (better quality) or "local" (offline, free, fast)
# Use "local" if you don't want to spend Gemini API quota on embeddings
EMBEDDING_PROVIDER=local

# ChromaDB settings
CHROMA_PERSIST_DIR=data/chroma_db
CHROMA_COLLECTION_NAME=indian_laws

# Chunking settings for the PDF pipeline
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

---

## 🗄️ Running the Data Pipeline

> **This is a one-time setup step.** The pipeline reads all 26 PDFs, chunks them, generates embeddings, and stores them in ChromaDB. Once done, the `data/chroma_db/` folder persists the database — you never need to run this again unless you add new PDFs.

### Step 1 — Verify Your PDFs

Make sure all 26 PDF files are inside `data/Acts/`. You can run a quick analysis:

```bash
python check_data.py
```

This will print a report of all PDFs — page count, character count, estimated chunks, and whether Gemini's free tier is sufficient for your data.

### Step 2 — Run the Ingestion Pipeline

```bash
python backend/scripts/ingest_pdfs.py
```

**What this does:**
1. Scans all 26 PDF files in `data/Acts/`
2. Extracts and cleans text using PyMuPDF (handles complex legal formatting)
3. Splits text into overlapping chunks (1000 chars, 200 overlap) with metadata (Act name, page range)
4. Generates vector embeddings using your configured provider (Gemini or local)
5. Stores all vectors in ChromaDB at `data/chroma_db/`
6. Saves chunk JSON files to `data/chunks/` for inspection
7. Runs a smoke test query to verify retrieval is working

**Optional flags:**

```bash
# Reset the ChromaDB collection before re-ingesting (use if you want a clean slate)
python backend/scripts/ingest_pdfs.py --reset

# Use a smaller batch size if you hit memory issues
python backend/scripts/ingest_pdfs.py --batch-size 50

# Skip saving chunk JSON files (faster, saves disk space)
python backend/scripts/ingest_pdfs.py --no-save-chunks
```

**Expected output:**
```
================================================================================
VAKALAT AI -- DATA INGESTION PIPELINE (PHASE 0)
================================================================================
Found 26 Act PDFs to process.

[ 1/26] [OK] Arms ACT                                  | 144 pages |  364 chunks | 3.12s
[ 2/26] [OK] BNS                                       | 232 pages |  618 chunks | 5.47s
...
[ 26/26] [OK] Wildlife Protection Act                  | 187 pages |  354 chunks | 3.89s

Total Chunks Extracted: 12,239
Generating embeddings & indexing into ChromaDB...
[OK] Embeddings indexed in 142.30 seconds.

ChromaDB Documents: 12,239
Pipeline execution finished successfully!
```

---

## 🖥️ Running the Backend Server

Once the data pipeline is complete and your `.env` is configured, start the API server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

| Flag | Purpose |
|------|---------|
| `--reload` | Auto-restarts the server when you save code changes (dev only) |
| `--host 0.0.0.0` | Makes the server accessible from other devices on your network (required for Flutter on real devices) |
| `--port 8000` | Port to run on |

Once running, open your browser:

| URL | Purpose |
|-----|---------|
| `http://localhost:8000/health` | Health check — should return `{"status": "healthy"}` |
| `http://localhost:8000/docs` | Interactive Swagger UI — test all endpoints in the browser |
| `http://localhost:8000/redoc` | ReDoc API documentation |

---

## 🔌 API Reference

### Endpoints Overview

| Method | Endpoint | Content-Type | Description |
|--------|----------|--------------|-------------|
| `GET` | `/health` | — | Server health check |
| `POST` | `/api/v1/offense/identify` | `application/json` | Text description → legal offense JSON |
| `POST` | `/api/v1/offense/identify-voice` | `multipart/form-data` | Voice file → legal offense JSON + audio |
| `POST` | `/api/v1/audio/transcribe` | `multipart/form-data` | Voice file → transcribed text |
| `POST` | `/api/v1/audio/synthesize` | `application/x-www-form-urlencoded` | Text → streamable MP3 audio |

---

### `POST /api/v1/offense/identify`

Accepts a plain-language description of an incident and returns a structured legal analysis.

**Request Body:**
```json
{
  "query": "Someone hacked into my email account and is blackmailing me with my private photos",
  "language": "en",
  "filter_act": null
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | `string` | ✅ Yes | User's description of the incident (min 5 characters) |
| `language` | `string` | No | Response language — `"en"` (default) or `"hi"` |
| `filter_act` | `string` | No | Restrict retrieval to a specific Act (e.g. `"BNS"`, `"POCSO"`, `"IT Act"`) |

**Response:**
```json
{
  "primary_offense": "Cyber Blackmail and Voyeurism",
  "primary_act": "Information Technology Act, 2000",
  "sections": [
    {
      "section_number": "Section 66E",
      "section_title": "Violation of privacy",
      "punishment": "Imprisonment up to 3 years or fine up to Rs. 2,00,000 or both",
      "is_bailable": false,
      "is_cognizable": true
    },
    {
      "section_number": "Section 67A",
      "section_title": "Punishment for publishing sexually explicit material",
      "punishment": "Imprisonment up to 5 years and fine up to Rs. 10,00,000 on first conviction",
      "is_bailable": false,
      "is_cognizable": true
    }
  ],
  "plain_explanation": "Sharing or threatening to share someone's private images without consent is a punishable offense under the IT Act. The accused can be arrested without a warrant.",
  "recommended_action": "File a complaint at your nearest Cyber Crime police station or online at cybercrime.gov.in. Preserve all evidence (screenshots, messages).",
  "confidence_score": 0.93,
  "source_citations": ["IT ACT (Pages 23,24)", "IT ACT (Pages 31,32)"],
  "audio_url": null
}
```

---

### `POST /api/v1/audio/transcribe`

Accepts a voice recording and returns the transcribed text. Supports `.m4a`, `.aac`, `.wav`, `.mp3`, `.ogg`.

**Request:** `multipart/form-data` with field `file`

**Response:**
```json
{
  "text": "Someone hacked into my email and is threatening me",
  "detected_language": "en"
}
```

---

### `POST /api/v1/audio/synthesize`

Converts text to natural-sounding Indian speech.

**Request:** `application/x-www-form-urlencoded`

| Field | Value |
|-------|-------|
| `text` | The text to convert to speech |
| `language` | `"en"` (Indian English) or `"hi"` (Hindi) |

**Response:** Raw MP3 audio bytes (`Content-Type: audio/mpeg`)

---

### Error Responses

| Status Code | When |
|-------------|------|
| `422 Unprocessable Entity` | Invalid request body / missing required fields |
| `500 Internal Server Error` | AI chain failure or ChromaDB error |

---

## 🧪 Running Tests

### Retrieval Accuracy Test (Phase 0 — already works ✅)

Validates that ChromaDB correctly retrieves relevant Acts for 7 legal categories:

```bash
python backend/tests/test_retrieval.py
```

Expected output:
```
TEST SUITE: VAKALAT AI -- RETRIEVAL ACCURACY & RELEVANCE
Total Test Cases: 7
...
Pass Rate: 100.0%
CHECKPOINT 0.5 PASSED: Retrieval accuracy meets quality standards!
```

### Offense Chain Test (Phase 1)

Runs 20+ real-world citizen scenarios through the full AI chain:

```bash
python backend/tests/test_offense_chain.py
```

### Audio Pipeline Test (Phase 1)

Tests the STT and TTS services with sample audio:

```bash
python backend/tests/test_audio_pipeline.py
```

---

## 🗓️ Development Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 0** | Data pipeline — PDF ingestion, chunking, ChromaDB vector store | ✅ **Complete** |
| **Phase 1** | Building Version 1 (Offense Identifier AI) — FastAPI backend, RAG chain, audio I/O | 🔄 **In Progress** |
| **Phase 2** | Building Version 2 (Law Tutor & Grader AI) — explain, quiz, grade student answers | 🔲 Planned |
| **Phase 3** | Polish, testing & production deployment | 🔲 Planned |

For the full sprint-by-sprint breakdown with tasks, checkpoints, and acceptance criteria, see [`ROADMAP.md`](./ROADMAP.md) and [`PHASE_1_GUIDE.md`](./PHASE_1_GUIDE.md).

---

## 🤝 Contributing

This project is in active development. The codebase is not yet open for external contributions. Once Phase 1 reaches its first stable release, contribution guidelines will be published.

---

## 📄 License

This project is **private and proprietary**. All rights reserved.

Unauthorized copying, distribution, or use of any part of this codebase is strictly prohibited.

---

<div align="center">

**Built with ❤️ for India's Legal Community**

*Making the law accessible to every citizen — in their own language*

</div>
