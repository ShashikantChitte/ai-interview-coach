"""Two-stage safety layer (PART 6).

Stage 1: rule-based blocklist — fast, zero cost, catches obvious cases.
Stage 2: model-based classifier — only fires when Stage 1 passes.
"""

from __future__ import annotations

import logging

from .config import MODEL_NAME, SAFETY_TEMPERATURE
from .prompts import SAFETY_CLASSIFIER_PROMPT

logger = logging.getLogger(__name__)


JAILBREAK_PATTERNS: tuple[str, ...] = (
    "ignore previous instructions",
    "ignore your instructions",
    "forget your system prompt",
    "act as",
    "pretend you are",
    "you are now",
    "jailbreak",
    "dan mode",
)

MIN_MEANINGFUL_LENGTH: int = 5


def rule_based_safety_check(user_input: str) -> bool:
    """Return True if the input passes the rule-based blocklist."""
    lowered = user_input.lower()

    for pattern in JAILBREAK_PATTERNS:
        if pattern in lowered:
            logger.warning("[Safety] Rule blocked: matched %r", pattern)
            return False

    if len(user_input.strip()) < MIN_MEANINGFUL_LENGTH:
        logger.warning("[Safety] Rule blocked: input too short")
        return False

    return True


def model_based_safety_check(client, user_input: str) -> bool:
    """Return True if the model classifies the input as SAFE.

    In mock mode this returns True without an API call (the mock client
    is flagged with `_is_mock = True`).
    """
    if getattr(client, "_is_mock", False):
        return True

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SAFETY_CLASSIFIER_PROMPT},
            {"role": "user", "content": user_input},
        ],
        max_tokens=5,
        temperature=SAFETY_TEMPERATURE,
    )
    verdict = (response.choices[0].message.content or "").strip().upper()

    if verdict != "SAFE":
        logger.warning("[Safety] Model blocked input. Verdict: %s", verdict)
        return False
    return True


def safety_check(client, user_input: str) -> bool:
    """Full two-stage safety check. Returns True if safe."""
    if not rule_based_safety_check(user_input):
        return False
    return model_based_safety_check(client, user_input)
