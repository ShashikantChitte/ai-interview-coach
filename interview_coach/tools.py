"""PART 4 - Tool definition + Tool Arbiter (dispatcher)."""
from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from .schemas import AnswerScore


class ScoringError(ValueError):
    """Raised when the model's tool arguments fail validation."""


def clean_schema(node: Any) -> Any:
    """Strip Pydantic's cosmetic "title" keys; some providers reject them."""
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key == "properties" and isinstance(value, dict):
                out[key] = {name: clean_schema(sub) for name, sub in value.items()}
            elif key == "title" and isinstance(value, str):
                continue
            else:
                out[key] = clean_schema(value)
        return out
    if isinstance(node, list):
        return [clean_schema(v) for v in node]
    return node


def build_tool_definition() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "score_answer",
            "description": "Score a candidate's interview answer on Clarity, Correctness and Completeness. Call after every answer.",
            "parameters": clean_schema(AnswerScore.model_json_schema()),
        },
    }


SCORING_TOOL = build_tool_definition()


def handle_score_answer(arguments: dict) -> AnswerScore:
    try:
        return AnswerScore(**arguments)
    except ValidationError as exc:
        raise ScoringError(f"Model returned invalid scoring JSON: {exc}") from exc


# Dispatch table: add new tools here and nowhere else.
HANDLERS = {"score_answer": handle_score_answer}


def tool_arbiter(tool_name: str, arguments_json: str, extra: dict | None = None) -> str:
    """Route a tool call to its handler and return a JSON string for the `tool` message.

    `extra` is merged into the result (used to pass `next_step` instructions back to the model).
    """
    handler = HANDLERS.get(tool_name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    result = handler(json.loads(arguments_json))
    payload = result.model_dump()
    if extra:
        payload.update(extra)
    return json.dumps(payload)
