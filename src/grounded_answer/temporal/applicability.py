"""Apply an ingested policy change to original clause text."""

from __future__ import annotations

from grounded_answer.domain.clause import PolicyClause
from grounded_answer.domain.policy_change import ChangeType, PolicyChange


def apply_change(original: PolicyClause | None, change: PolicyChange) -> str:
    if change.change_type is ChangeType.INSERT:
        return change.new_rule
    if original is None:
        return change.new_rule
    if change.change_type is ChangeType.REPLACE:
        return _replace_table(original.content, change.new_rule)
    if change.previous_rule and change.previous_rule in original.content:
        return original.content.replace(change.previous_rule, change.new_rule)
    return original.content


def _replace_table(original: str, new_table: str) -> str:
    index = original.find("|")
    if index == -1:
        return f"{original.rstrip()}\n\n{new_table.strip()}"
    return original[:index] + new_table.strip()
