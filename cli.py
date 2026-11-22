"""Interactive mock interview in your terminal.

    python cli.py --role "Product Manager"
"""
import argparse
import logging

from interview_coach import InterviewSession


def print_score(score) -> None:
    print(f"   [score] Clarity={score.clarity} | Correctness={score.correctness} | "
          f"Completeness={score.completeness} | Pass={score.pass_threshold}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Interview Coach")
    parser.add_argument("--role", default="Software Engineer")
    parser.add_argument("--questions", type=int, default=3)
    parser.add_argument("-v", "--verbose", action="store_true", help="show token + safety logs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")

    session = InterviewSession(role=args.role, max_questions=args.questions)
    print("=" * 60, "\nAI INTERVIEW COACH  (type 'quit' to exit)\n" + "=" * 60)
    print(f"\n[COACH]\n{session.start()}")

    while session.interview_active:
        try:
            text = input("\n[YOU]\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"quit", "exit"}:
            break
        before = len(session.scores)
        reply = session.answer(text)
        if len(session.scores) > before:
            print_score(session.scores[-1])
        print(f"\n[COACH]\n{reply}")

    print("\n" + "=" * 60, "\nSESSION SUMMARY\n" + "=" * 60)
    for key, value in session.get_summary().items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
