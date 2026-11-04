"""Pydantic schema for structured scoring (PART 3)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .config import MAX_SCORE, MIN_SCORE


class AnswerScore(BaseModel):
    """Structured evaluation of a single interview answer.

    Field order matters: `reasoning` is declared first so the model
    generates its Chain-of-Thought trace *before* the numeric scores.
    This conditions the scores on a completed reasoning chain.
    """

    reasoning: str = Field(
        description=(
            "Step-by-step analysis of the answer before assigning scores. "
            "Think through what the answer got right, what it missed, and why."
        )
    )
    clarity: int = Field(
        ge=MIN_SCORE,
        le=MAX_SCORE,
        description=(
            "How clearly was the answer communicated? "
            "1=incoherent, 5=perfectly structured"
        ),
    )
    correctness: int = Field(
        ge=MIN_SCORE,
        le=MAX_SCORE,
        description=(
            "How accurate was the answer? "
            "1=fundamentally wrong, 5=completely correct"
        ),
    )
    completeness: int = Field(
        ge=MIN_SCORE,
        le=MAX_SCORE,
        description=(
            "How complete was the answer? "
            "1=major gaps, 5=covered everything important"
        ),
    )
    feedback: str = Field(
        description=(
            "2-3 sentences of specific, actionable feedback. "
            "Do not give away the correct answer. Point to gaps."
        )
    )
    pass_threshold: bool = Field(
        description=(
            "True if the overall answer meets a passing bar "
            "(average score >= 3). False otherwise."
        )
    )
