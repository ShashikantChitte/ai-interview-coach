"""PART 2 + 3 - CoT scoring schema (Pydantic).

Field order matters: `reasoning` comes first so the model writes its analysis
BEFORE committing to numeric scores (chain-of-thought inside structured output).
"""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class AnswerScore(BaseModel):
    reasoning: str = Field(
        description=(
            "Step-by-step analysis of the answer BEFORE assigning scores. "
            "Think through what the answer got right, what it missed, and why."
        )
    )
    clarity: int = Field(ge=1, le=5, description="How clearly was the answer communicated? 1=incoherent, 5=perfectly structured")
    correctness: int = Field(ge=1, le=5, description="How accurate was the answer? 1=fundamentally wrong, 5=completely correct")
    completeness: int = Field(ge=1, le=5, description="How complete was the answer? 1=major gaps, 5=covered everything important")
    feedback: str = Field(
        description="2-3 sentences of specific, actionable feedback. Do NOT give away the correct answer. Point to gaps."
    )
    pass_threshold: bool = Field(
        description="True if the average of the three scores is >= 3, otherwise False."
    )

    @property
    def average(self) -> float:
        return (self.clarity + self.correctness + self.completeness) / 3

    @model_validator(mode="after")
    def _enforce_pass_rule(self) -> "AnswerScore":
        # Don't trust the model to do arithmetic: derive the flag deterministically.
        self.pass_threshold = self.average >= 3
        return self
