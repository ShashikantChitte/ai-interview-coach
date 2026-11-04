"""Mock OpenAI client — lets the demo run without an API key.

The mock mimics the shape of `openai.OpenAI().chat.completions.create(...)`
responses so the rest of the code runs unchanged.
"""

from __future__ import annotations

from typing import Any


# ---- Response shapes that mimic the OpenAI SDK ----

class MockFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class MockToolCall:
    def __init__(self, id: str, name: str, arguments: str) -> None:
        self.id = id
        self.type = "function"
        self.function = MockFunction(name, arguments)


class MockMessage:
    def __init__(self, content: str | None = None, tool_calls: list[MockToolCall] | None = None) -> None:
        self.role = "assistant"
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in self.tool_calls
            ]
        if exclude_none:
            data = {k: v for k, v in data.items() if v is not None}
        return data


class MockChoice:
    def __init__(self, message: MockMessage) -> None:
        self.message = message


class MockUsage:
    def __init__(self, total_tokens: int = 150) -> None:
        self.total_tokens = total_tokens


class MockResponse:
    def __init__(self, message: MockMessage, total_tokens: int = 150) -> None:
        self.choices = [MockChoice(message)]
        self.usage = MockUsage(total_tokens)


# ---- Canned responses for a 3-question interview ----

_SCORE_1 = (
    '{"reasoning": "Candidate described a concrete conflict about code review '
    'standards and a structured resolution: 1:1, active listening, shared checklist. '
    'Missing: how they handled disagreement about the technical merits specifically.", '
    '"clarity": 4, "correctness": 4, "completeness": 3, '
    '"feedback": "Good use of a concrete conflict and a structured resolution. '
    'You did not explain how you resolved the technical disagreement itself — only '
    'the process around it. Next time, walk through the specific technical tradeoff."'
    ', "pass_threshold": true}'
)

_SCORE_2 = (
    '{"reasoning": "Candidate correctly identified O(1) average lookup, hashing to '
    'array index, and both chaining and open addressing for collisions. No major '
    'gaps. Could add load-factor and worst-case O(n) caveat.", '
    '"clarity": 5, "correctness": 5, "completeness": 4, '
    '"feedback": "Excellent. Consider adding the worst-case complexity under heavy '
    'collision and how load factor triggers resizing.", '
    '"pass_threshold": true}'
)

_SCORE_3 = (
    '{"reasoning": "Candidate identified a queue (correct) but did not mention BFS, '
    'level separators, or tracking level size. Answer is correct in direction but '
    'incomplete.", '
    '"clarity": 3, "correctness": 4, "completeness": 2, '
    '"feedback": "A queue is the right primitive, but you did not explain BFS or how '
    'to separate levels. Mention tracking current-level size or a sentinel.", '
    '"pass_threshold": false}'
)


class MockCompletions:
    def __init__(self) -> None:
        self._index = 0
        self._canned = self._build()

    def _build(self) -> list[MockResponse]:
        return [
            MockResponse(MockMessage(content=(
                "Here's your first question: Tell me about a time you had a technical "
                "disagreement with a teammate. How did you handle it?"
            )), total_tokens=180),
            MockResponse(MockMessage(tool_calls=[
                MockToolCall("call_1", "score_answer", _SCORE_1)
            ]), total_tokens=240),
            MockResponse(MockMessage(content=(
                "Clarity 4/5, Correctness 4/5, Completeness 3/5. You described a clear "
                "resolution process but did not explain how you resolved the technical "
                "disagreement itself.\n"
                "Next question: Explain how a hashmap handles collisions."
            )), total_tokens=220),
            MockResponse(MockMessage(tool_calls=[
                MockToolCall("call_2", "score_answer", _SCORE_2)
            ]), total_tokens=240),
            MockResponse(MockMessage(content=(
                "Clarity 5/5, Correctness 5/5, Completeness 4/5. Excellent technical "
                "answer. Consider adding the worst-case complexity under heavy collisions.\n"
                "Final question: How would you traverse a binary tree level by level?"
            )), total_tokens=220),
            MockResponse(MockMessage(tool_calls=[
                MockToolCall("call_3", "score_answer", _SCORE_3)
            ]), total_tokens=240),
            MockResponse(MockMessage(content=(
                "Clarity 3/5, Correctness 4/5, Completeness 2/5. A queue is correct, "
                "but you did not explain BFS or how to separate levels.\n"
                "That concludes our session. Your summary is available via get_summary()."
            )), total_tokens=220),
        ]

    def create(self, **kwargs: Any) -> MockResponse:
        # Safety classifier call — short max_tokens
        if kwargs.get("max_tokens") == 5:
            return MockResponse(MockMessage(content="SAFE"), total_tokens=10)

        if self._index >= len(self._canned):
            return MockResponse(MockMessage(content="Session complete."), total_tokens=20)

        resp = self._canned[self._index]
        self._index += 1
        return resp


class MockChat:
    def __init__(self) -> None:
        self.completions = MockCompletions()


class MockClient:
    """Drop-in stand-in for `openai.OpenAI()`."""

    _is_mock = True

    def __init__(self) -> None:
        self.chat = MockChat()
