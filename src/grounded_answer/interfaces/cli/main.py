"""CLI entry point: python -m grounded_answer ask \"QUESTION\"."""

from __future__ import annotations

import argparse
import sys

from grounded_answer.interfaces.cli.commands import ask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grounded_answer",
        description="Ask a question against the Household Support Program policy.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    ask_parser = subparsers.add_parser("ask", help="Answer a policy question")
    ask_parser.add_argument("question", help="Question to ask")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    if args.command == "ask":
        sys.stdout.write(ask(args.question))
        return 0
    return 1
