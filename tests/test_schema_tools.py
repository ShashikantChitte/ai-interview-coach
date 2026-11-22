import json

import pytest
from pydantic import ValidationError

from interview_coach.schemas import AnswerScore
from interview_coach.tools import SCORING_TOOL, ScoringError, tool_arbiter

GOOD = {"reasoning": "r", "clarity": 5, "correctness": 2, "completeness": 2, "feedback": "f", "pass_threshold": True}


def test_out_of_range_score_rejected():
    with pytest.raises(ValidationError):
        AnswerScore(**{**GOOD, "clarity": 0})


def test_pass_flag_is_computed_not_trusted():
    low = AnswerScore(**{**GOOD, "clarity": 1, "correctness": 2, "completeness": 2, "pass_threshold": True})
    assert low.pass_threshold is False  # model claimed True, average is 1.67


def test_pass_flag_boundary_is_inclusive():
    assert AnswerScore(**GOOD).average == 3.0
    assert AnswerScore(**{**GOOD, "pass_threshold": False}).pass_threshold is True


def test_reasoning_is_first_field():
    assert list(AnswerScore.model_fields)[0] == "reasoning"
    assert list(SCORING_TOOL["function"]["parameters"]["properties"])[0] == "reasoning"


def test_tool_schema_has_no_titles_and_keeps_bounds():
    params = SCORING_TOOL["function"]["parameters"]
    assert "title" not in params
    assert all("title" not in p for p in params["properties"].values())
    assert params["properties"]["clarity"]["minimum"] == 1
    assert params["properties"]["clarity"]["maximum"] == 5


def test_arbiter_dispatch_and_extra():
    out = json.loads(tool_arbiter("score_answer", json.dumps(GOOD), {"next_step": "go"}))
    assert out["next_step"] == "go" and out["clarity"] == 5


def test_arbiter_unknown_tool():
    assert "error" in json.loads(tool_arbiter("nope", "{}"))


def test_arbiter_invalid_args():
    with pytest.raises(ScoringError):
        tool_arbiter("score_answer", json.dumps({**GOOD, "clarity": 9}))
