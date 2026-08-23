"""CLI entry point: python -m grounded_answer ask \"QUESTION\"."""

from __future__ import annotations

import argparse
import sys
from datetime import date

from grounded_answer.interfaces.cli.commands import ask


def _parse_iso_date(value: str) -> date:
    return date.fromisoformat(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grounded_answer",
        description="Ask a question against the Household Support Program policy.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    ask_parser = subparsers.add_parser("ask", help="Answer a policy question")
    ask_parser.add_argument("question", help="Question to ask")
    ask_parser.add_argument(
        "--determination-date",
        type=_parse_iso_date,
        default=None,
        help="Determination date (YYYY-MM-DD)",
    )
    ask_parser.add_argument(
        "--change-of-circumstances-date",
        type=_parse_iso_date,
        default=None,
        help="Date the change of circumstances occurred (YYYY-MM-DD)",
    )
    ask_parser.add_argument(
        "--claim-start-date",
        type=_parse_iso_date,
        default=None,
        help="Claim period start (YYYY-MM-DD)",
    )
    ask_parser.add_argument(
        "--claim-end-date",
        type=_parse_iso_date,
        default=None,
        help="Claim period end (YYYY-MM-DD)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    if args.command == "ask":
        sys.stdout.write(
            ask(
                args.question,
                determination_date=args.determination_date,
                change_of_circumstances_date=args.change_of_circumstances_date,
                claim_start_date=args.claim_start_date,
                claim_end_date=args.claim_end_date,
            )
        )
        return 0
    return 1
