"""Programmatic demo — import and run from anywhere.

    python -m examples.demo_session
"""

from src.interview_session import InterviewSession
from src.mock_client import MockClient


def main() -> None:
    session = InterviewSession(client=MockClient(), role="Software Engineer")

    print("=" * 60)
    print("AI INTERVIEW COACH — Programmatic Demo (mock)")
    print("=" * 60)

    print("\n[COACH]:", session.start())

    for answer in [
        "At my last internship I had a conflict with a teammate about code review "
        "standards. I scheduled a 1:1, listened to their concerns, and we agreed on "
        "a shared checklist.",
        "A hashmap stores key-value pairs. Lookup is O(1) average because you hash "
        "the key to get the array index directly. Collisions are handled with "
        "chaining or open addressing.",
        "I would use a queue. You process nodes level by level.",
    ]:
        print("\n[CANDIDATE]:", answer)
        print("\n[COACH]:", session.answer(answer))

    summary = session.get_summary()
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
