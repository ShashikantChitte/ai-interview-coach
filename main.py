"""CLI entry point for the AI Interview Coach.

Usage:
    python main.py --demo --mock         # run scripted demo without API
    python main.py --demo                # run scripted demo with API
    python main.py --interactive         # interactive Q&A session
    python main.py --interactive --role "Product Manager"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from src.interview_session import InterviewSession


def _build_client(mock: bool):
    if mock:
        from src.mock_client import MockClient
        print("[main] Using MOCK client (no API calls).")
        return MockClient()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "ERROR: OPENAI_API_KEY is not set.\n"
            "  Option A: add it to your .env file\n"
            "  Option B: run with --mock to use canned responses",
            file=sys.stderr,
        )
        sys.exit(1)

    from openai import OpenAI
    print("[main] Using live OpenAI client.")
    return OpenAI(api_key=api_key)


def _print_summary(session: InterviewSession) -> None:
    summary = session.get_summary()
    print("\n" + "=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\nPer-question scores:")
    for i, score in enumerate(session.scores, 1):
        print(
            f"\n  Q{i}: Clarity={score.clarity} | "
            f"Correctness={score.correctness} | "
            f"Completeness={score.completeness} | "
            f"Pass={score.pass_threshold}"
        )
        print(f"  Feedback: {score.feedback}")


def run_demo(client) -> None:
    session = InterviewSession(client=client, role="Software Engineer")

    print("=" * 60)
    print("AI INTERVIEW COACH — Live Session")
    print("=" * 60)

    print("\n[COACH]:")
    print(session.start())

    # Q1 — behavioral
    print("\n" + "-" * 60)
    print("[CANDIDATE]:")
    answer_1 = (
        "At my last internship I had a conflict with a teammate about code review "
        "standards. I scheduled a 1:1, listened to their concerns, and we agreed on "
        "a shared checklist. It worked well."
    )
    print(answer_1)
    print("\n[COACH]:")
    print(session.answer(answer_1))

    # Q2 — technical (hashmap)
    print("\n" + "-" * 60)
    print("[CANDIDATE]:")
    answer_2 = (
        "A hashmap stores key-value pairs. Lookup is O(1) average because you hash "
        "the key to get the array index directly. Collisions are handled with "
        "chaining or open addressing."
    )
    print(answer_2)
    print("\n[COACH]:")
    print(session.answer(answer_2))

    # Q3 — incomplete answer to test scoring
    print("\n" + "-" * 60)
    print("[CANDIDATE]:")
    answer_3 = "I would use a queue. You process nodes level by level."
    print(answer_3)
    print("\n[COACH]:")
    print(session.answer(answer_3))

    _print_summary(session)


def run_interactive(client, role: str) -> None:
    session = InterviewSession(client=client, role=role)

    print("=" * 60)
    print(f"AI INTERVIEW COACH — Interactive ({role})")
    print("Type your answers. Type 'quit' to exit early.")
    print("=" * 60)

    print("\n[COACH]:")
    print(session.start())

    while session.interview_active:
        try:
            user = input("\n[CANDIDATE]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[main] Exiting.")
            break

        if user.lower() in {"quit", "exit"}:
            print("[main] Ending session early.")
            break

        reply = session.answer(user)
        print("\n[COACH]:")
        print(reply)

    _print_summary(session)


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Interview Coach")
    parser.add_argument("--demo", action="store_true",
                        help="Run the scripted demo")
    parser.add_argument("--interactive", action="store_true",
                        help="Run interactively")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock client (no API)")
    parser.add_argument("--role", default="Software Engineer", help="Job role")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    if not args.demo and not args.interactive:
        parser.print_help()
        sys.exit(0)

    client = _build_client(args.mock)

    if args.demo:
        run_demo(client)
    else:
        run_interactive(client, args.role)


if __name__ == "__main__":
    main()
