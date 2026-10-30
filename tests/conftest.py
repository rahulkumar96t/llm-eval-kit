from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "LLM_EVAL_MODEL",
        "LLM_EVAL_JUDGE_MODEL",
        "LLM_EVAL_JUDGE_EFFORT",
        "LLM_EVAL_FALLBACKS",
        "LLM_EVAL_CACHE_DIR",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def examples_dir() -> Path:
    return EXAMPLES


@pytest.fixture
def write_suite(tmp_path: Path):
    """Write a suite YAML plus a JSONL dataset into tmp_path and return the suite path."""

    def _write(suite_yaml: str, rows: list[str], name: str = "suite.yaml") -> Path:
        (tmp_path / "data.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
        path = tmp_path / name
        path.write_text(suite_yaml, encoding="utf-8")
        return path

    return _write
