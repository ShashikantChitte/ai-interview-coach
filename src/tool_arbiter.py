"""Tool definition and Tool Arbiter (PART 4)."""

from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from .schemas import AnswerScore

logger = logging.getLogger(__name__)


def build_tool_definition() -> dict:
    """Build the OpenAI function-calling tool definition from the Pydantic schema."""
    schema = AnswerScore.model_json_schema()
    return {
        "type": "function",
        "function": {
            "name": "score_answer",
            "description": (
                "Score a candidate's interview answer on Clarity, Correctness, "
                "and Completeness. Call this after every answer."
            ),
            "parameters": schema,
        },
    }


SCORING_TOOL: dict = build_tool_definition()


def handle_score_answer(arguments: dict) -> AnswerScore:
    """Parse and validate the model's scoring output."""
    try:
        return AnswerScore(**arguments)
    except ValidationError as exc:
        raise ValueError(
            f"Model returned invalid scoring JSON: {exc}") from exc


def tool_arbiter(tool_name: str, arguments_json: str) -> str:
    """Dispatch a tool call and return a serialized result string."""
    arguments = json.loads(arguments_json)

    if tool_name == "score_answer":
        result = handle_score_answer(arguments)
        return result.model_dump_json()

    logger.error("[Tool] Unknown tool: %s", tool_name)
    return json.dumps({"error": f"Unknown tool: {tool_name}"})
