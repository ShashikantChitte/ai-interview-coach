"""The complete interaction loop (PART 7)."""

from __future__ import annotations

import json
import logging
from typing import Any

from .config import (
    MAX_HISTORY_TOKENS,
    MAX_QUESTIONS,
    MODEL_NAME,
    PASS_QUESTIONS_REQUIRED,
    TEMPERATURE,
)
from .context_manager import trim_messages
from .prompts import INTERVIEW_SYSTEM_PROMPT
from .safety import safety_check
from .schemas import AnswerScore
from .tool_arbiter import SCORING_TOOL, tool_arbiter

logger = logging.getLogger(__name__)


class InterviewSession:
    """Runs a full interview: safety → API → tool handling → summary."""

    def __init__(self, client, role: str = "Software Engineer",
                 max_questions: int = MAX_QUESTIONS) -> None:
        self.client = client
        self.role = role
        self.max_questions = max_questions
        self.messages: list[dict] = [
            {
                "role": "system",
                "content": INTERVIEW_SYSTEM_PROMPT.format(max_questions=max_questions),
            }
        ]
        self.question_count = 0
        self.scores: list[AnswerScore] = []
        self.interview_active = True
        self.total_tokens_used = 0

    # ---- internal API helper ----

    def _call_api(self, include_tools: bool = True) -> Any:
        trimmed = trim_messages(self.messages, MAX_HISTORY_TOKENS)

        kwargs: dict[str, Any] = {
            "model": MODEL_NAME,
            "messages": trimmed,
            "temperature": TEMPERATURE,
        }
        if include_tools:
            kwargs["tools"] = [SCORING_TOOL]
            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)

        usage = response.usage
        self.total_tokens_used += usage.total_tokens
        logger.info(
            "[Tokens] This call: %d | Session total: %d",
            usage.total_tokens, self.total_tokens_used,
        )
        return response

    def _assistant_message_dict(self, message: Any) -> dict:
        """Serialize an assistant message (real or mock) to a history dict."""
        data: dict[str, Any] = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            data["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
        return {k: v for k, v in data.items() if v is not None}

    def _handle_tool_calls(self, response: Any) -> str:
        message = response.choices[0].message

        if not message.tool_calls:
            return message.content or ""

        self.messages.append(self._assistant_message_dict(message))

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments
            logger.info("[Tool] Calling: %s", tool_name)

            result_str = tool_arbiter(tool_name, tool_args)

            if tool_name == "score_answer":
                score_data = json.loads(result_str)
                self.scores.append(AnswerScore(**score_data))

            self.messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_str,
            })

        follow_up = self._call_api(include_tools=False)
        return follow_up.choices[0].message.content or ""

    # ---- public API ----

    def start(self) -> str:
        self.messages.append({
            "role": "user",
            "content": (
                f"I'm ready to be interviewed for a {self.role} role. Please begin."
            ),
        })
        response = self._call_api()
        reply = response.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": reply})
        self.question_count += 1
        return reply

    def answer(self, candidate_answer: str) -> str:
        if not self.interview_active:
            return "The interview session has ended. Call get_summary() for results."

        if not safety_check(self.client, candidate_answer):
            return (
                "That input is not related to the interview. "
                "Please answer the current question."
            )

        self.messages.append({"role": "user", "content": candidate_answer})
        response = self._call_api()
        reply = self._handle_tool_calls(response)
        self.messages.append({"role": "assistant", "content": reply})

        if self.question_count < self.max_questions:
            self.question_count += 1
        else:
            self.interview_active = False

        return reply

    def get_summary(self) -> dict:
        if not self.scores:
            return {"error": "No scores recorded yet."}

        n = len(self.scores)
        avg_clarity = sum(s.clarity for s in self.scores) / n
        avg_correctness = sum(s.correctness for s in self.scores) / n
        avg_completeness = sum(s.completeness for s in self.scores) / n
        questions_passed = sum(1 for s in self.scores if s.pass_threshold)

        return {
            "questions_asked": n,
            "average_clarity": round(avg_clarity, 1),
            "average_correctness": round(avg_correctness, 1),
            "average_completeness": round(avg_completeness, 1),
            "questions_passed": questions_passed,
            "overall_result": (
                "PASS"
                if questions_passed >= PASS_QUESTIONS_REQUIRED
                else "FAIL"
            ),
            "total_tokens_used": self.total_tokens_used,
        }