"""Resolve which policy version applies before the LLM is called."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from grounded_answer.domain.amendment import Amendment, AmendmentParagraph
from grounded_answer.domain.clause import PolicyClause
from grounded_answer.domain.evidence import Evidence
from grounded_answer.domain.policy import Policy
from grounded_answer.domain.policy_change import ApplicabilityBasis, ChangeType, PolicyChange
from grounded_answer.domain.temporal import TemporalContext
from grounded_answer.temporal.applicability import apply_change
from grounded_answer.temporal.timeline import format_period, split_claim_period

APPORTION_CLAUSE_ID = "§7.4.3"


@dataclass(frozen=True, slots=True)
class ApplicablePolicyRule:
    clause_id: str
    rule_text: str
    source: str
    effective_from: date | None
    effective_until: date | None
    reasoning: str


@dataclass(frozen=True, slots=True)
class ResolutionResult:
    evidence: tuple[Evidence, ...]
    rules: tuple[ApplicablePolicyRule, ...]
    unresolved_targets: tuple[str, ...]
    conflict: bool


class PolicyApplicabilityResolver:
    """Choose applicable clause text using amendment rules and temporal context.

    The LLM is not asked to pick between conflicting versions.
    """

    def __init__(self, policy: Policy, amendment: Amendment) -> None:
        self._policy = policy
        self._amendment = amendment
        self._clauses = {clause.clause_id: clause for clause in policy.clauses}
        self._paragraphs = {item.paragraph_id: item for item in amendment.paragraphs}
        self._changes = {change.target_clause: change for change in amendment.changes}

    def resolve(
        self,
        retrieved: Sequence[Evidence],
        temporal: TemporalContext,
    ) -> ResolutionResult:
        retrieved_ids = {item.clause_id for item in retrieved}
        targets = self._candidate_targets(retrieved_ids)
        evidence: list[Evidence] = []
        rules: list[ApplicablePolicyRule] = []
        unresolved: list[str] = []
        seen: set[str] = set()
        used_paragraphs: set[str] = set()
        conflict = False

        for clause_id in targets:
            change = self._changes.get(clause_id)
            original = self._clauses.get(clause_id)
            if change is None:
                if original is None or clause_id in seen:
                    continue
                evidence.append(_from_original(original))
                seen.add(clause_id)
                continue
            resolved = self._resolve_change(original, change, temporal)
            if resolved.conflict:
                conflict = True
                continue
            if resolved.unresolved:
                unresolved.append(clause_id)
                continue
            if resolved.omit:
                continue
            for item in resolved.items:
                key = f"{item.clause_id}:{item.applicable_period}:{item.policy_version}"
                if key in seen:
                    continue
                seen.add(key)
                evidence.append(item)
            rules.extend(resolved.rules)
            used_paragraphs.update(resolved.paragraph_ids)

        for item in retrieved:
            if item.clause_id in self._clauses and not any(
                existing.clause_id == item.clause_id for existing in evidence
            ):
                if item.clause_id in unresolved or item.clause_id in self._changes:
                    continue
                evidence.append(
                    Evidence(
                        evidence_id=item.evidence_id,
                        clause_id=item.clause_id,
                        content=item.content,
                        source=item.source,
                        policy_version=item.policy_version,
                        applicable_period=item.applicable_period,
                        applicability_basis=item.applicability_basis or ApplicabilityBasis.ORIGINAL.value,
                    )
                )

        for paragraph_id in used_paragraphs:
            paragraph = self._paragraphs.get(paragraph_id)
            if paragraph is None:
                continue
            if any(item.clause_id == paragraph_id for item in evidence):
                continue
            evidence.append(_from_paragraph(paragraph))

        return ResolutionResult(
            evidence=tuple(evidence),
            rules=tuple(rules),
            unresolved_targets=tuple(unresolved),
            conflict=conflict,
        )

    def clause_content(self, clause_id: str) -> str:
        clause = self._clauses.get(clause_id)
        if clause is not None:
            return clause.content
        change = self._changes.get(clause_id)
        if change is not None:
            return change.new_rule
        paragraph = self._paragraphs.get(clause_id)
        if paragraph is not None:
            return paragraph.content
        return ""

    def known_ids(self) -> set[str]:
        ids = {clause.clause_id for clause in self._policy.clauses}
        ids.update(self._paragraphs)
        ids.update(self._changes)
        return ids

    def _candidate_targets(self, retrieved_ids: set[str]) -> list[str]:
        targets: list[str] = []

        def add(clause_id: str) -> None:
            if clause_id not in targets:
                targets.append(clause_id)

        for clause_id in retrieved_ids:
            if clause_id in self._clauses or clause_id in self._changes:
                add(clause_id)
        for change in self._amendment.changes:
            if change.change_id in retrieved_ids:
                add(change.target_clause)
        if any(item.startswith("§10.5") for item in retrieved_ids):
            for change in self._amendment.changes:
                if change.change_type is ChangeType.INSERT:
                    add(change.target_clause)
        return targets

    def _resolve_change(
        self,
        original: PolicyClause | None,
        change: PolicyChange,
        temporal: TemporalContext,
    ) -> _ClauseResolution:
        effective = self._amendment.effective_date
        if change.applicability is ApplicabilityBasis.CHANGE_OF_CIRCUMSTANCES_DATE:
            relevant = temporal.change_of_circumstances_date
            if relevant is None:
                return _ClauseResolution(unresolved=True)
            amended = relevant >= effective
            return self._single_version(
                original,
                change,
                amended,
                relevant,
                ApplicabilityBasis.CHANGE_OF_CIRCUMSTANCES_DATE,
                extra_paragraphs=("¶5.2",),
            )

        span = self._spanning_period(temporal, change)
        if span is not None:
            return span

        relevant = temporal.determination_date
        if relevant is None:
            relevant = self._in_force_proxy(temporal)
        if relevant is None:
            return _ClauseResolution(unresolved=True)
        amended = relevant >= effective
        return self._single_version(
            original,
            change,
            amended,
            relevant,
            ApplicabilityBasis.DETERMINATION_DATE,
            extra_paragraphs=("¶5.1",),
        )

    def _spanning_period(
        self,
        temporal: TemporalContext,
        change: PolicyChange,
    ) -> _ClauseResolution | None:
        start = temporal.claim_start_date
        end = temporal.claim_end_date
        if start is None or end is None:
            return None
        if end < start:
            return _ClauseResolution(conflict=True)
        effective = self._amendment.effective_date
        parts = split_claim_period(start, end, effective)
        if len(parts) != 2:
            return None
        if change.section_number not in {1, 3}:
            return None
        original = self._clauses.get(change.target_clause)
        original_text = original.content if original else ""
        amended_text = apply_change(original, change)
        source = original.source_document if original else change.source
        blocks = [
            (
                "Applicable figures differ across 1 March 2026. "
                "The award is apportioned under §7.4.3 by reference to the number of days."
            )
        ]
        for part_start, part_end, version in parts:
            text = original_text if version == "original" else amended_text
            label = "original" if version == "original" else f"amended by {change.change_id}"
            blocks.append(f"Period {format_period(part_start, part_end)} ({label}):\n{text}")
        combined = "\n\n".join(blocks)
        items = [
            Evidence(
                evidence_id=f"{source}:{change.target_clause}:transitional",
                clause_id=change.target_clause,
                content=combined,
                source=source,
                policy_version="transitional",
                applicable_period=format_period(start, end),
                applicability_basis=ApplicabilityBasis.CROSS_BOUNDARY.value,
            )
        ]
        apportion = self._clauses.get(APPORTION_CLAUSE_ID)
        if apportion:
            items.append(_from_original(apportion))
        reasoning = (
            f"basis=cross_boundary; claim={format_period(start, end)}; "
            f"effective={effective.isoformat()}; change={change.change_id}; apportion={APPORTION_CLAUSE_ID}"
        )
        rule = ApplicablePolicyRule(
            clause_id=change.target_clause,
            rule_text=combined,
            source=source,
            effective_from=start,
            effective_until=end,
            reasoning=reasoning,
        )
        return _ClauseResolution(
            items=tuple(items),
            rules=(rule,),
            paragraph_ids=("¶5.3",),
        )

    def _single_version(
        self,
        original: PolicyClause | None,
        change: PolicyChange,
        amended: bool,
        relevant: date,
        basis: ApplicabilityBasis,
        extra_paragraphs: tuple[str, ...],
    ) -> _ClauseResolution:
        effective = self._amendment.effective_date
        if change.change_type is ChangeType.INSERT and not amended:
            return _ClauseResolution(omit=True)
        if amended:
            text = apply_change(original, change)
            source = change.source if change.change_type is ChangeType.INSERT else (
                original.source_document if original else change.source
            )
            version = "amended"
            paragraph_ids = extra_paragraphs
            effective_from = effective
            effective_until = None
        else:
            if original is None:
                return _ClauseResolution(omit=True)
            text = original.content
            source = original.source_document
            version = "original"
            paragraph_ids = extra_paragraphs
            effective_from = None
            effective_until = date.fromordinal(effective.toordinal() - 1)
        reasoning = (
            f"basis={basis.value}; date={relevant.isoformat()}; "
            f"effective={effective.isoformat()}; version={version}; change={change.change_id}"
        )
        item = Evidence(
            evidence_id=f"{source}:{change.target_clause}:{version}",
            clause_id=change.target_clause,
            content=text,
            source=source,
            policy_version=version,
            applicable_period=relevant.isoformat(),
            applicability_basis=basis.value,
        )
        rule = ApplicablePolicyRule(
            clause_id=change.target_clause,
            rule_text=text,
            source=source,
            effective_from=effective_from,
            effective_until=effective_until,
            reasoning=reasoning,
        )
        return _ClauseResolution(items=(item,), rules=(rule,), paragraph_ids=paragraph_ids)

    def _in_force_proxy(self, temporal: TemporalContext) -> date | None:
        start = temporal.claim_start_date
        end = temporal.claim_end_date
        if start is None:
            return None
        if end is None:
            end = start
        effective = self._amendment.effective_date
        if start < effective <= end:
            return None
        return start


@dataclass(frozen=True, slots=True)
class _ClauseResolution:
    items: tuple[Evidence, ...] = ()
    rules: tuple[ApplicablePolicyRule, ...] = ()
    paragraph_ids: tuple[str, ...] = ()
    unresolved: bool = False
    omit: bool = False
    conflict: bool = False


def _from_original(clause: PolicyClause) -> Evidence:
    return Evidence(
        evidence_id=f"{clause.source_document}:{clause.clause_id}:original",
        clause_id=clause.clause_id,
        content=clause.content,
        source=clause.source_document,
        policy_version="original",
        applicability_basis=ApplicabilityBasis.ORIGINAL.value,
    )


def _from_paragraph(paragraph: AmendmentParagraph) -> Evidence:
    return Evidence(
        evidence_id=f"{paragraph.source_document}:{paragraph.paragraph_id}",
        clause_id=paragraph.paragraph_id,
        content=paragraph.content,
        source=paragraph.source_document,
        policy_version="amendment",
        applicability_basis="amendment_provision",
    )
