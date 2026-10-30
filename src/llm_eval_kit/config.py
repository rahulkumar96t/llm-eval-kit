"""Suite configuration (YAML) and dataset loading (JSONL)."""

from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
)

from .models import EFFORT_LEVELS

REMOVED_SAMPLING_PARAMS = {"temperature", "top_p", "top_k"}


class ConfigError(Exception):
    """Raised for invalid suites, datasets or templates."""


class ModelParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_tokens: int = Field(default=16000, gt=0)
    effort: str | None = None
    stop_sequences: list[str] = Field(default_factory=list)

    @field_validator("effort")
    @classmethod
    def _check_effort(cls, value: str | None) -> str | None:
        if value is None or value == "none":
            return None
        if value not in EFFORT_LEVELS:
            raise ValueError(f"effort must be one of {', '.join(EFFORT_LEVELS)} or 'none'")
        return value


class ScorerSpec(BaseModel):
    """One scorer entry. Type-specific options are validated by the scorer itself."""

    model_config = ConfigDict(extra="allow")

    type: str
    name: str | None = None

    @property
    def options(self) -> dict[str, Any]:
        return dict(self.model_extra or {})

    @property
    def display_name(self) -> str:
        return self.name or self.type


class ScorerThreshold(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pass_rate: float | None = Field(default=None, ge=0, le=1)
    mean_score: float | None = Field(default=None, ge=0, le=1)


class Thresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pass_rate: float = Field(default=1.0, ge=0, le=1)
    scorers: dict[str, ScorerThreshold] = Field(default_factory=dict)
    max_regressions: int = Field(default=0, ge=0)
    score_tolerance: float = Field(default=0.0, ge=0, le=1)


class Suite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    dataset: Path
    provider: Literal["anthropic", "recorded"] = "anthropic"
    output_field: str = "output"
    model: str | None = None
    system: str | None = None
    prompt: str = ""
    params: ModelParams = Field(default_factory=ModelParams)
    scorers: list[ScorerSpec] = Field(default_factory=list)
    thresholds: Thresholds = Field(default_factory=Thresholds)

    # Set by load_suite: the directory relative paths are resolved against.
    _base_dir: Path = PrivateAttr(default=Path("."))

    @field_validator("params", mode="before")
    @classmethod
    def _reject_sampling(cls, value: Any) -> Any:
        if isinstance(value, dict):
            removed = REMOVED_SAMPLING_PARAMS & value.keys()
            if removed:
                raise ValueError(
                    f"{', '.join(sorted(removed))} cannot be sent: Claude API version "
                    "2026-10-21 rejects sampling parameters on every model"
                )
        return value

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    @property
    def dataset_path(self) -> Path:
        return self.dataset if self.dataset.is_absolute() else self.base_dir / self.dataset

    def resolve(self, relative: str | Path) -> Path:
        path = Path(relative)
        return path if path.is_absolute() else self.base_dir / path


class Case(BaseModel):
    """One dataset row. Every field is available to the prompt template."""

    model_config = ConfigDict(extra="forbid")

    id: str
    vars: dict[str, Any]
    scorers: list[ScorerSpec] = Field(default_factory=list)


def load_suite(path: str | Path) -> Suite:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"suite file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: a suite must be a YAML mapping")
    try:
        suite = Suite.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"{path}: {_format_errors(exc)}") from exc
    suite._base_dir = path.parent
    if suite.provider == "anthropic" and not suite.prompt:
        raise ConfigError(f"{path}: 'prompt' is required when provider is 'anthropic'")
    return suite


def load_cases(suite: Suite) -> list[Case]:
    path = suite.dataset_path
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ConfigError(f"dataset not found: {path}") from exc
    cases: list[Case] = []
    seen: set[str] = set()
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
        if not isinstance(row, dict):
            raise ConfigError(f"{path}:{lineno}: each line must be a JSON object")
        case_id = str(row.pop("id", f"case-{lineno:04d}"))
        if case_id in seen:
            raise ConfigError(f"{path}:{lineno}: duplicate case id {case_id!r}")
        seen.add(case_id)
        try:
            scorers = [ScorerSpec.model_validate(s) for s in row.pop("scorers", [])]
        except ValidationError as exc:
            raise ConfigError(f"{path}:{lineno}: {_format_errors(exc)}") from exc
        cases.append(Case(id=case_id, vars=row, scorers=scorers))
    if not cases:
        raise ConfigError(f"{path}: dataset is empty")
    return cases


def render(template: str, variables: dict[str, Any], *, where: str = "template") -> str:
    """Fill ``{field}`` placeholders; ``{{`` and ``}}`` are literal braces."""
    fields = {name for _, name, _, _ in string.Formatter().parse(template) if name}
    missing = sorted(f for f in fields if f.split(".")[0].split("[")[0] not in variables)
    if missing:
        raise ConfigError(f"{where}: missing field(s) {', '.join(missing)}")
    try:
        return template.format_map(variables)
    except (ValueError, IndexError, KeyError, AttributeError) as exc:
        raise ConfigError(f"{where}: cannot render ({exc})") from exc


def _format_errors(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors():
        loc = ".".join(str(p) for p in error["loc"]) or "(root)"
        parts.append(f"{loc}: {error['msg']}")
    return "; ".join(parts)
