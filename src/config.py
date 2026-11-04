"""Central configuration for the AI Interview Coach."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# --- Model & API ---
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.3"))
SAFETY_TEMPERATURE: float = 0.0  # deterministic safety decisions

# --- Token budget ---
CONTEXT_LIMIT: int = int(os.getenv("CONTEXT_LIMIT", "8000"))
COMPLETION_RESERVE: int = int(os.getenv("COMPLETION_RESERVE", "1000"))
MAX_HISTORY_TOKENS: int = CONTEXT_LIMIT - COMPLETION_RESERVE

# --- Session ---
MAX_QUESTIONS: int = int(os.getenv("MAX_QUESTIONS", "3"))

# --- Scoring ---
MIN_SCORE: int = 1
MAX_SCORE: int = 5
PASS_AVERAGE_THRESHOLD: float = 3.0
PASS_QUESTIONS_REQUIRED: int = 2  # out of MAX_QUESTIONS to pass overall
