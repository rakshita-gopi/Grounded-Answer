# Grounded Answer

A CLI-first system that answers questions about Calder County's Household Support Program policy.

Answers must be grounded in the supplied policy corpus, include citations to the supporting clauses, and abstain when the corpus does not support an answer.

## Current status

The original policy corpus is in `data/policy/policy-manual.md`. Core domain models live under `src/grounded_answer/domain/`. Application services are not implemented yet, and the project is not runnable.

## Prerequisites

- Python 3.11 or later (to be confirmed when dependencies are added)

## Installation

Dependencies will be added as features are implemented. See `requirements.txt`.
