import json
from types import SimpleNamespace

import pytest

from interview_coach.config import Config


def make_resp(content=None, tool_calls=None, tokens=10):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=SimpleNamespace(total_tokens=tokens))


class FakeClient:
    """Offline stand-in for the OpenAI client."""

    def __init__(self, scores=(4, 4, 4), classifier="SAFE"):
        self.scores = list(scores)
        self.classifier = classifier
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        if "safety classifier" in kw["messages"][0]["content"]:
            return make_resp(self.classifier)
        if kw.get("tool_choice") == "required":
            s = self.scores.pop(0)
            args = {"reasoning": "analysis", "clarity": s, "correctness": s, "completeness": s,
                    "feedback": "Be more specific.", "pass_threshold": False}
            tc = SimpleNamespace(id=f"call_{len(self.calls)}",
                                 function=SimpleNamespace(name="score_answer", arguments=json.dumps(args)))
            return make_resp(None, [tc])
        return make_resp("Question 1 of 3: Tell me about a conflict.")


@pytest.fixture
def cfg():
    return Config(provider="test", model="test-model", api_key="x", base_url=None)


@pytest.fixture
def fake():
    return FakeClient()
