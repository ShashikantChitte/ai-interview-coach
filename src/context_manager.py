"""Token-aware context management (PART 5)."""

from __future__ import annotations

import logging

import tiktoken

from .config import MODEL_NAME

logger = logging.getLogger(__name__)

PER_MESSAGE_OVERHEAD: int = 4
REPLY_PRIMER_TOKENS: int = 2


def _get_encoder():
    try:
        return tiktoken.encoding_for_model(MODEL_NAME)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(messages: list[dict]) -> int:
    """Count tokens in a message list using tiktoken."""
    enc = _get_encoder()
    total = 0
    for msg in messages:
        total += PER_MESSAGE_OVERHEAD
        for value in msg.values():
            if isinstance(value, str):
                total += len(enc.encode(value))
            elif isinstance(value, list):
                total += len(enc.encode(str(value)))
    total += REPLY_PRIMER_TOKENS
    return total


def trim_messages(messages: list[dict], max_tokens: int) -> list[dict]:
    """Sliding-window trim with pinned head (system) and pinned tail (last exchange)."""
    if count_tokens(messages) <= max_tokens:
        return messages

    system_msg = messages[0]
    tail = messages[-2:] if len(messages) >= 2 else messages[-1:]
    middle = messages[1:-2] if len(messages) > 3 else []

    while middle and count_tokens([system_msg] + middle + tail) > max_tokens:
        dropped = middle.pop(0)
        preview = str(dropped.get("content", ""))[:40]
        logger.info("[Context trim] Dropped message: %s (%s...)",
                    dropped.get("role"), preview)

    return [system_msg] + middle + tail
