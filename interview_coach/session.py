"""PART 7 - The interaction loop that wires everything together."""
from __future__ import annotations

import json
import logging

from .config import Config, get_client, load_config
from .context import trim_messages
from .prompts import OFF_TOPIC_SENTINEL, REFUSAL_MESSAGE, build_system_prompt
from .safety import safety_check
from .schemas import AnswerScore
from .tools import SCORING_TOOL, ScoringError, tool_arbiter

log = logging.getLogger(__name__)


class InterviewSession:
    def __init__(self, role: str = "Software Engineer", max_questions: int = 3,
                 client=None, config: Config | None = None):
        self.role = role
        self.max_questions = max_questions
        self.cfg = config or load_config()
        self.client = client or get_client(self.cfg)
        self.messages: list[dict] = [{"role": "system", "content": build_system_prompt(role, max_questions)}]
        self.scores: list[AnswerScore] = []
        self.interview_active = True
        self.total_tokens_used = 0

    # ---- API plumbing -------------------------------------------------
    def _chat(self, use_tools: bool):
        kwargs = dict(
            model=self.cfg.model,
            messages=trim_messages(self.messages, self.cfg.max_history_tokens),
            temperature=self.cfg.coach_temperature,
        )
        if use_tools:
            # Force scoring on every answer instead of hoping the model chooses to.
            kwargs.update(tools=[SCORING_TOOL], tool_choice="required")
        resp = self.client.chat.completions.create(**kwargs)
        usage = getattr(resp, "usage", None)
        if usage and usage.total_tokens:
            self.total_tokens_used += usage.total_tokens
            log.info("[Tokens] call=%s session=%s", usage.total_tokens, self.total_tokens_used)
        return resp

    def _next_step(self) -> str:
        done = len(self.scores) + 1  # the answer being scored right now
        if done >= self.max_questions:
            return (f"This was the final question ({done} of {self.max_questions}). Share the score and feedback, "
                    "then give a brief overall summary of strengths and areas to improve. Do NOT ask another question.")
        return (f"Share the score and 2-3 sentences of feedback in plain language, then ask question "
                f"{done + 1} of {self.max_questions}. Do not reveal the correct answer.")

    def _run_tool_calls(self, message) -> str:
        # Rebuild the assistant message explicitly for maximum provider compatibility.
        self.messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in message.tool_calls
            ],
        })
        for tc in message.tool_calls:
            result = tool_arbiter(tc.function.name, tc.function.arguments, {"next_step": self._next_step()})
            if tc.function.name == "score_answer":
                self.scores.append(AnswerScore(**json.loads(result)))
            self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        # Second call: the model now sees the tool result and writes the reply.
        return (self._chat(use_tools=False).choices[0].message.content or "").strip()

    # ---- Public API ---------------------------------------------------
    def start(self) -> str:
        self.messages.append({"role": "user", "content": f"I'm ready to be interviewed for a {self.role} role. Please begin."})
        reply = (self._chat(use_tools=False).choices[0].message.content or "").strip()
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def answer(self, candidate_answer: str) -> str:
        if not self.interview_active:
            return "The interview session has ended. Call get_summary() for results."
        if not safety_check(candidate_answer, self.client, self.cfg.model):
            return REFUSAL_MESSAGE

        checkpoint = len(self.messages)
        self.messages.append({"role": "user", "content": candidate_answer})
        try:
            message = self._chat(use_tools=True).choices[0].message
            reply = self._run_tool_calls(message) if message.tool_calls else (message.content or "").strip()
        except (ScoringError, json.JSONDecodeError) as exc:
            log.warning("Scoring failed: %s", exc)
            del self.messages[checkpoint:]  # roll back so history stays consistent
            return "Sorry, I couldn't evaluate that answer. Please send it again."

        if reply == OFF_TOPIC_SENTINEL:
            del self.messages[checkpoint:]
            return REFUSAL_MESSAGE

        self.messages.append({"role": "assistant", "content": reply})
        if len(self.scores) >= self.max_questions:
            self.interview_active = False
        return reply

    def get_summary(self) -> dict:
        """Deterministic summary computed locally from stored scores - no extra API call."""
        if not self.scores:
            return {"error": "No scores recorded yet."}
        n = len(self.scores)
        passed = sum(s.pass_threshold for s in self.scores)
        return {
            "role": self.role,
            "questions_scored": n,
            "average_clarity": round(sum(s.clarity for s in self.scores) / n, 1),
            "average_correctness": round(sum(s.correctness for s in self.scores) / n, 1),
            "average_completeness": round(sum(s.completeness for s in self.scores) / n, 1),
            "questions_passed": passed,
            "overall_result": "PASS" if passed >= 2 else "FAIL",
            "total_tokens_used": self.total_tokens_used,
        }
