"""
Vakalat AI — Configuration Module
Loads environment variables and centralized paths for Phase 0 and beyond.
"""
from pathlib import Path
import os
from dotenv import load_dotenv

# Base directories
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent

# Load .env file from project root or backend dir
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
ACTS_DIR = DATA_DIR / "Acts"
CHROMA_DIR = DATA_DIR / "chroma_db"
CHUNKS_DIR = DATA_DIR / "chunks"

# Ensure runtime directories exist
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

# Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Vector DB Configuration
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "indian_laws")

# API Keys and Embedding Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()

# Embedding Models
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
