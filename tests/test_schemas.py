"""Basic tests for the Pydantic scoring schema."""

import pytest
from pydantic import ValidationError

from src.schemas import AnswerScore


def _valid_payload(**overrides):
    base = dict(
        reasoning="Candidate explained the concept correctly.",
        clarity=4,
        correctness=5,
        completeness=3,
        feedback="Good, but could mention edge cases.",
        pass_threshold=True,
    )
    base.update(overrides)
    return base


def test_valid_score_constructs():
    score = AnswerScore(**_valid_payload())
    assert score.clarity == 4
    assert score.correctness == 5


def test_out_of_range_score_raises():
    with pytest.raises(ValidationError):
        AnswerScore(**_valid_payload(clarity=6))


def test_zero_score_raises():
    with pytest.raises(ValidationError):
        AnswerScore(**_valid_payload(correctness=0))


def test_reasoning_is_first_field_in_schema():
    schema = AnswerScore.model_json_schema()
    props = list(schema["properties"].keys())
    assert props[0] == "reasoning", (
        "reasoning must be declared first so the model generates its "
        "Chain-of-Thought trace before committing to numeric scores."
    )


def test_schema_enforces_bounds_in_json_schema():
    schema = AnswerScore.model_json_schema()
    clarity = schema["properties"]["clarity"]
    assert clarity["minimum"] == 1
    assert clarity["maximum"] == 5
