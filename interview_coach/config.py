"""Provider-agnostic configuration.

Every provider below exposes an OpenAI-compatible chat completions endpoint,
so one SDK (`openai`) talks to all of them. Switch with LLM_PROVIDER / LLM_MODEL.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

PROVIDERS: dict[str, dict] = {
    # Free tier via Google AI Studio. Flash-Lite is the cheapest Gemini tier.
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "model": "gemini-2.5-flash-lite",
    },
    # Free tier with rate limits.
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model": "llama-3.1-8b-instant",
    },
    # Fully local and free (install Ollama, then `ollama pull llama3.2`).
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "key_env": None,
        "model": "llama3.2",
    },
    # Paid but very cheap.
    "openai": {
        "base_url": None,
        "key_env": "OPENAI_API_KEY",
        "model": "gpt-4o-mini",
    },
}


@dataclass
class Config:
    provider: str
    model: str
    api_key: str
    base_url: str | None
    context_limit: int = 8000
    completion_reserve: int = 1000
    coach_temperature: float = 0.3

    @property
    def max_history_tokens(self) -> int:
        return self.context_limit - self.completion_reserve


def load_config() -> Config:
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Choose from: {', '.join(PROVIDERS)}")
    spec = PROVIDERS[provider]
    key_env = spec["key_env"]
    api_key = os.getenv(key_env, "") if key_env else "ollama"
    if not api_key:
        raise RuntimeError(f"Missing {key_env}. Copy .env.example to .env and add your key.")
    return Config(
        provider=provider,
        model=os.getenv("LLM_MODEL", spec["model"]),
        api_key=api_key,
        base_url=spec["base_url"],
        context_limit=int(os.getenv("CONTEXT_LIMIT", "8000")),
        completion_reserve=int(os.getenv("COMPLETION_RESERVE", "1000")),
    )


def get_client(cfg: Config):
    from openai import OpenAI

    return OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
