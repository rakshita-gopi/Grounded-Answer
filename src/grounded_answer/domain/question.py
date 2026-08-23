"""A user question to be answered from the policy corpus."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Question:
    text: str
