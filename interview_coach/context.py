"""PART 5 - Token-aware context management."""
from __future__ import annotations

import logging
from functools import lru_cache

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # offline / vocab download blocked
        return None


def _tok(text: str) -> int:
    enc = _encoder()
    return len(enc.encode(text)) if enc else len(text) // 4 + 1


def count_tokens(messages: list[dict]) -> int:
    """Estimate tokens in a message list.

    cl100k_base is exact for OpenAI models and a close estimate for others
    (Gemini, Llama), which is fine for budgeting. Falls back to ~4 chars/token.
    """
    total = 2  # reply primer
    for msg in messages:
        total += 4  # per-message overhead
        for value in msg.values():
            if isinstance(value, str):
                total += _tok(value)
            elif value is not None:
                total += _tok(str(value))
    return total


def _group_turns(rest: list[dict]) -> list[list[dict]]:
    """Group messages into turns that start at each user message.

    Dropping whole turns keeps assistant(tool_calls) and tool messages paired;
    orphaned tool messages make the API reject the request.
    """
    turns: list[list[dict]] = []
    for msg in rest:
        if msg["role"] == "user" or not turns:
            turns.append([msg])
        else:
            turns[-1].append(msg)
    return turns


def trim_messages(messages: list[dict], max_tokens: int) -> list[dict]:
    """Drop the oldest turns until under budget. Pins the system prompt and the latest turn."""
    if count_tokens(messages) <= max_tokens:
        return messages
    system, turns = messages[0], _group_turns(messages[1:])
    flat = lambda ts: [m for t in ts for m in t]  # noqa: E731
    while len(turns) > 1 and count_tokens([system] + flat(turns)) > max_tokens:
        dropped = turns.pop(0)
        log.info("[Context trim] dropped %d message(s) starting with role=%s", len(dropped), dropped[0]["role"])
    return [system] + flat(turns)
