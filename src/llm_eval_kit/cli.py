"""Command-line interface: ``llm-eval run | diff | validate``."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NoReturn

from . import __version__
from .baseline import BaselineDiff, diff_runs
from .cache import ResponseCache
from .config import ConfigError, Suite, load_cases, load_suite
from .models import AnthropicModelClient, ModelClient, fallbacks_enabled
from .ratelimit import RetryPolicy, TokenBucket
from .reports import render_html, render_junit, render_markdown, write_report
from .results import CaseResult, RunResult
from .runner import ModelGateway, Runner

EXIT_OK = 0
EXIT_THRESHOLD = 1  # bit flag
EXIT_REGRESSION = 2  # bit flag; 3 = thresholds failed and regressions found
EXIT_CONFIG = 4
EXIT_RUNTIME = 5

ClientFactory = Callable[[], ModelClient]


class _Parser(argparse.ArgumentParser):
    """argparse exits with 2 on usage errors; 2 means 'regression' here, so remap."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(EXIT_CONFIG, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="llm-eval", description="Regression tests for LLM apps.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, parser_class=_Parser)

    run = sub.add_parser("run", help="run a suite and score the outputs")
    run.add_argument("suite", type=Path, help="suite YAML file")
    run.add_argument("--model", help="model under test (overrides the suite and LLM_EVAL_MODEL)")
    run.add_argument("--concurrency", type=int, default=4, help="parallel cases (default 4)")
    run.add_argument("--rpm", type=float, help="max model requests per minute (default: no limit)")
    run.add_argument("--max-attempts", type=int, default=4, help="attempts per request (default 4)")
    run.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("LLM_EVAL_CACHE_DIR", ".llm-eval-cache")),
        help="response cache directory (default .llm-eval-cache or $LLM_EVAL_CACHE_DIR)",
    )
    run.add_argument("--no-cache", action="store_true", help="neither read nor write the cache")
    run.add_argument("--output", type=Path, help="write the run as JSON")
    run.add_argument("--baseline", type=Path, help="compare against this stored run")
    run.add_argument(
        "--allow-missing-baseline",
        action="store_true",
        help="skip the comparison instead of failing when --baseline does not exist",
    )
    run.add_argument("--save-baseline", type=Path, help="store this run as the new baseline")
    run.add_argument("--tolerance", type=float, help="allowed score drop before a regression")
    run.add_argument("--report-md", type=Path, help="write a Markdown report")
    run.add_argument("--report-html", type=Path, help="write a self-contained HTML report")
    run.add_argument("--junit", type=Path, help="write JUnit XML")
    run.add_argument("-q", "--quiet", action="store_true", help="no per-case progress lines")

    diff = sub.add_parser("diff", help="compare two stored runs")
    diff.add_argument("baseline", type=Path)
    diff.add_argument("current", type=Path)
    diff.add_argument("--tolerance", type=float, default=0.0)
    diff.add_argument("--max-regressions", type=int, default=0)

    validate = sub.add_parser("validate", help="check a suite and its dataset without model calls")
    validate.add_argument("suite", type=Path)
    return parser


def main(argv: Sequence[str] | None = None, client_factory: ClientFactory | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return asyncio.run(_cmd_run(args, client_factory or AnthropicModelClient))
        if args.command == "diff":
            return _cmd_diff(args)
        return _cmd_validate(args)
    except ConfigError as exc:
        print(f"llm-eval: config error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except OSError as exc:
        print(f"llm-eval: {exc}", file=sys.stderr)
        return EXIT_CONFIG


def _progress(case: CaseResult) -> None:
    label = {"passed": "PASS", "failed": "FAIL", "error": "ERR "}[case.status]
    note = " (cached)" if case.cached else ""
    print(f"[{label}] {case.id} {case.latency_ms:.0f} ms{note}", file=sys.stderr)


async def _cmd_run(args: argparse.Namespace, client_factory: ClientFactory) -> int:
    suite = load_suite(args.suite)
    cases = load_cases(suite)
    if args.concurrency < 1 or args.max_attempts < 1:
        raise ConfigError("--concurrency and --max-attempts must be >= 1")
    baseline = _load_baseline(args)

    cache = None if args.no_cache else ResponseCache(args.cache_dir)
    limiter = TokenBucket.per_minute(args.rpm) if args.rpm else None
    retry = RetryPolicy(max_attempts=args.max_attempts)
    client = client_factory()
    gateway = ModelGateway(
        client,
        cache=cache,
        limiter=limiter,
        retry=retry,
        cache_extra={"fallbacks": getattr(client, "fallbacks", fallbacks_enabled())},
    )
    runner = Runner(
        suite,
        model_gateway=gateway if suite.provider == "anthropic" else None,
        judge_gateway=gateway,
        concurrency=args.concurrency,
        model_override=args.model,
        on_case_done=None if args.quiet else _progress,
    )
    run = await runner.run(cases)

    tolerance = args.tolerance if args.tolerance is not None else suite.thresholds.score_tolerance
    diff = diff_runs(baseline, run, tolerance=tolerance) if baseline else None
    _write_outputs(args, run, diff)
    _print_summary(run, diff)
    if run.summary.errors == run.summary.total and run.summary.total > 0:
        print("llm-eval: every case errored - check credentials and model id", file=sys.stderr)
        return EXIT_RUNTIME
    return _exit_code(run, diff, suite)


def _load_baseline(args: argparse.Namespace) -> RunResult | None:
    if not args.baseline:
        return None
    if not args.baseline.exists():
        if args.allow_missing_baseline:
            print(f"llm-eval: no baseline at {args.baseline}; skipping comparison", file=sys.stderr)
            return None
        raise ConfigError(f"baseline not found: {args.baseline}")
    try:
        return RunResult.load(args.baseline)
    except ValueError as exc:
        raise ConfigError(f"cannot read baseline {args.baseline}: {exc}") from exc


def _write_outputs(args: argparse.Namespace, run: RunResult, diff: BaselineDiff | None) -> None:
    if args.output:
        run.save(args.output)
    if args.save_baseline:
        run.save(args.save_baseline)
    if args.report_md:
        write_report(args.report_md, render_markdown(run, diff))
    if args.report_html:
        write_report(args.report_html, render_html(run, diff))
    if args.junit:
        write_report(args.junit, render_junit(run, diff))


def _print_summary(run: RunResult, diff: BaselineDiff | None) -> None:
    s = run.summary
    print(
        f"{run.suite}: {s.passed}/{s.total} passed ({s.pass_rate:.1%}), {s.errors} errors, "
        f"{s.cache_hits} cached, {run.duration_s:.1f} s"
    )
    for check in run.thresholds:
        status = "ok" if check.ok else "FAILED"
        print(f"  threshold {check.name}: {check.actual:.3f} >= {check.required:.3f} {status}")
    if diff is not None:
        print(
            f"  vs baseline: {len(diff.regressions)} regressions, "
            f"{len(diff.improvements)} improvements, {len(diff.new_cases)} new, "
            f"{len(diff.removed_cases)} removed"
        )
        for change in diff.regressions:
            print(f"    REGRESSION {change.case_id}: {change.reason}")


def _exit_code(run: RunResult, diff: BaselineDiff | None, suite: Suite) -> int:
    code = EXIT_OK
    if not all(check.ok for check in run.thresholds):
        code |= EXIT_THRESHOLD
    if diff is not None and len(diff.regressions) > suite.thresholds.max_regressions:
        code |= EXIT_REGRESSION
    return code


def _cmd_diff(args: argparse.Namespace) -> int:
    try:
        baseline, current = RunResult.load(args.baseline), RunResult.load(args.current)
    except ValueError as exc:
        raise ConfigError(f"cannot read run file: {exc}") from exc
    diff = diff_runs(baseline, current, tolerance=args.tolerance)
    _print_summary(current, diff)
    return EXIT_REGRESSION if len(diff.regressions) > args.max_regressions else EXIT_OK


def _cmd_validate(args: argparse.Namespace) -> int:
    suite = load_suite(args.suite)
    cases = load_cases(suite)
    Runner(suite, model_gateway=None).plan(cases)
    print(f"{suite.name}: ok ({len(cases)} cases, provider {suite.provider})")
    return EXIT_OK
