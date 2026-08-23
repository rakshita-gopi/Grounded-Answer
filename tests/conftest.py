from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def corpus_dir(repo_root: Path) -> Path:
    return repo_root / "data" / "policy"


@pytest.fixture
def sample_policy_path(repo_root: Path) -> Path:
    return repo_root / "tests" / "fixtures" / "sample_policy.md"
