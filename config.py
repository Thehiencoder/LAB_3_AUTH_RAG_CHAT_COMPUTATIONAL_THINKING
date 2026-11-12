"""Read settings from .env; never hardcode secrets or server URLs in source code."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Use .env NEXT TO this file, even if the app was started from another folder.
# Variables already set in the terminal/hosting environment take precedence.
load_dotenv(Path(__file__).with_name(".env"), override=False)

FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "").strip()
FIREBASE_API_KEY = os.getenv("FIREBASE_API_KEY", "").strip()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "").strip().rstrip("/")

EMBED_MODEL = os.getenv("EMBED_MODEL", "embeddinggemma").strip() or "embeddinggemma"
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3:4b").strip() or "qwen3:4b"
TOP_K = int(os.getenv("TOP_K", "3") or "3")
MIN_SIMILARITY_SCORE = float(os.getenv("MIN_SIMILARITY_SCORE", "0.5") or "0.5")
