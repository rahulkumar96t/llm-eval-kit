"""Report writers: Markdown, self-contained HTML and JUnit XML."""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from .baseline import BaselineDiff
from .results import CaseResult, RunResult

STATUS_ICON = {"passed": "PASS", "failed": "FAIL", "error": "ERROR"}

# Characters that are not allowed anywhere in an XML 1.0 document.
_XML_INVALID = re.compile("[^\x09\x0a\x0d\x20-\ud7ff\ue000-\ufffd\U00010000-\U0010ffff]")


def _xml_safe(text: str) -> str:
    return _XML_INVALID.sub("\ufffd", text)


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _md_cell(text: str, limit: int = 120) -> str:
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text.replace("|", "\\|") or " "


def _fence(text: str) -> str:
    """A code fence longer than any backtick run inside ``text``."""
    longest, run = 0, 0
    for char in text:
        run = run + 1 if char == "`" else 0
        longest = max(longest, run)
    return "`" * max(3, longest + 1)


def render_markdown(run: RunResult, diff: BaselineDiff | None = None) -> str:
    s = run.summary
    lines = [
        f"# Eval report: {run.suite}",
        "",
        f"- Provider / model: `{run.provider}` / `{run.model}`",
        f"- Started: {run.started_at} ({run.duration_s:.1f} s)",
        f"- Cases: {s.total} total, {s.passed} passed, {s.failed} failed, {s.errors} errors "
        f"({_pct(s.pass_rate)} pass rate)",
        f"- Cache hits: {s.cache_hits}; tokens in/out: {s.input_tokens}/{s.output_tokens}",
        "",
        "## Thresholds",
        "",
        "| Check | Actual | Required | Result |",
        "| --- | --- | --- | --- |",
    ]
    for check in run.thresholds:
        result = "ok" if check.ok else "**FAILED**"
        lines.append(
            f"| {check.name} | {check.actual:.3f} | {check.comparison} {check.required:.3f} "
            f"| {result} |"
        )
    lines += ["", "## Scorers", "", "| Scorer | Pass rate | Mean score | Cases |"]
    lines.append("| --- | --- | --- | --- |")
    for scorer in s.scorers:
        lines.append(
            f"| {scorer.name} | {_pct(scorer.pass_rate)} | {scorer.mean_score:.3f} "
            f"| {scorer.count} |"
        )
    if diff is not None:
        lines += ["", "## Baseline comparison", ""]
        lines.append(
            f"Pass rate {_pct(diff.pass_rate_before)} -> {_pct(diff.pass_rate_after)}; "
            f"{len(diff.regressions)} regressions, {len(diff.improvements)} improvements, "
            f"{len(diff.new_cases)} new, {len(diff.removed_cases)} removed, "
            f"{diff.unchanged} unchanged."
        )
        sections = (("Regressions", diff.regressions), ("Improvements", diff.improvements))
        for title, changes in sections:
            if changes:
                lines += ["", f"### {title}", ""]
                lines += [f"- `{c.case_id}`: {_md_cell(c.reason, 200)}" for c in changes]
    failing = [c for c in run.cases if not c.passed]
    lines += ["", "## Cases", "", "| Case | Status | Scores | Detail |"]
    lines.append("| --- | --- | --- | --- |")
    for case in run.cases:
        scores = ", ".join(f"{sc.name}={sc.score:.2f}" for sc in case.scores) or "-"
        detail = case.error or "; ".join(sc.detail for sc in case.scores if not sc.passed)
        status = STATUS_ICON[case.status]
        lines.append(f"| `{case.id}` | {status} | {_md_cell(scores)} | {_md_cell(detail)} |")
    if failing:
        lines += ["", "## Failing outputs", ""]
        for case in failing:
            output = case.output or "(no output)"
            fence = _fence(output)
            lines += [f"### `{case.id}`", "", fence, output, fence, ""]
    return "\n".join(lines).rstrip() + "\n"


_CSS = """
:root { --bg:#fbfbfa; --fg:#1d1d1b; --muted:#6b6b66; --line:#e2e1dc; --card:#ffffff;
  --pass:#1f7a4d; --fail:#b42318; --err:#9a5b00; --code:#f3f2ee; }
@media (prefers-color-scheme: dark) { :root { --bg:#151514; --fg:#ecebe6; --muted:#a3a29b;
  --line:#33332f; --card:#1d1d1b; --pass:#5cc08b; --fail:#ff8a7a; --err:#e5b45c; --code:#262624; } }
* { box-sizing: border-box; }
body { margin:0; padding:24px 16px; background:var(--bg); color:var(--fg);
  font:15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 1.5rem; margin: 0 0 4px; } h2 { font-size: 1.1rem; margin: 28px 0 8px; }
.meta { color: var(--muted); margin: 0 0 16px; }
.cards { display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px; }
.card b { display:block; font-size:1.4rem; font-variant-numeric: tabular-nums; }
.card span { color: var(--muted); font-size: .85rem; }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; background: var(--card); }
th, td { border-bottom: 1px solid var(--line); padding: 6px 10px; text-align: left;
  vertical-align: top; }
th { font-size: .8rem; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }
td.num { font-variant-numeric: tabular-nums; }
.passed { color: var(--pass); font-weight: 600; } .failed { color: var(--fail); font-weight: 600; }
.error { color: var(--err); font-weight: 600; }
details { margin: 4px 0; } summary { cursor: pointer; }
pre { background: var(--code); padding: 10px; border-radius: 6px; white-space: pre-wrap;
  word-break: break-word; margin: 6px 0; font-size: .85rem; }
"""


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def _case_row(case: CaseResult) -> str:
    scores = "".join(
        f"<div><span class='{'passed' if sc.passed else 'failed'}'>{_esc(sc.name)}</span> "
        f"{sc.score:.2f}{(' - ' + _esc(sc.detail)) if sc.detail else ''}</div>"
        for sc in case.scores
    )
    error = f"<div class='error'>{_esc(case.error)}</div>" if case.error else ""
    body = (
        f"<details><summary>prompt / output</summary>"
        f"<pre>{_esc(case.prompt or '(none)')}</pre><pre>{_esc(case.output or '(no output)')}</pre>"
        f"</details>"
    )
    return (
        f"<tr><td><code>{_esc(case.id)}</code></td>"
        f"<td class='{case.status}'>{STATUS_ICON[case.status]}</td>"
        f"<td>{scores}{error}{body}</td>"
        f"<td class='num'>{case.latency_ms:.0f} ms{' (cached)' if case.cached else ''}</td></tr>"
    )


def render_html(run: RunResult, diff: BaselineDiff | None = None) -> str:
    """A single self-contained file: inline CSS, no scripts, every value escaped."""
    s = run.summary
    cards = [
        (_pct(s.pass_rate), "pass rate"),
        (f"{s.passed}/{s.total}", "cases passed"),
        (str(s.errors), "errors"),
        (str(s.cache_hits), "cache hits"),
        (f"{run.duration_s:.1f} s", "duration"),
    ]
    if diff is not None:
        cards.append((str(len(diff.regressions)), "regressions vs baseline"))
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>Eval report: {_esc(run.suite)}</title><style>{_CSS}</style></head><body><main>",
        f"<h1>Eval report: {_esc(run.suite)}</h1>",
        f"<p class='meta'>{_esc(run.provider)} / {_esc(run.model)} - started {_esc(run.started_at)}"
        f" - tokens in/out {s.input_tokens}/{s.output_tokens}</p>",
        "<div class='cards'>",
        *(f"<div class='card'><b>{_esc(v)}</b><span>{_esc(k)}</span></div>" for v, k in cards),
        "</div><h2>Thresholds</h2><div class='table-wrap'><table>",
        "<tr><th>Check</th><th>Actual</th><th>Required</th><th>Result</th></tr>",
    ]
    for check in run.thresholds:
        cls, label = ("passed", "ok") if check.ok else ("failed", "failed")
        parts.append(
            f"<tr><td>{_esc(check.name)}</td><td class='num'>{check.actual:.3f}</td>"
            f"<td class='num'>{check.comparison} {check.required:.3f}</td>"
            f"<td class='{cls}'>{label}</td></tr>"
        )
    parts.append("</table></div><h2>Scorers</h2><div class='table-wrap'><table>")
    parts.append("<tr><th>Scorer</th><th>Pass rate</th><th>Mean score</th><th>Cases</th></tr>")
    for scorer in s.scorers:
        parts.append(
            f"<tr><td>{_esc(scorer.name)}</td><td class='num'>{_pct(scorer.pass_rate)}</td>"
            f"<td class='num'>{scorer.mean_score:.3f}</td><td class='num'>{scorer.count}</td></tr>"
        )
    parts.append("</table></div>")
    if diff is not None:
        parts.append("<h2>Baseline comparison</h2><div class='table-wrap'><table>")
        parts.append("<tr><th>Case</th><th>Change</th><th>Reason</th></tr>")
        for change in diff.regressions + diff.improvements:
            cls = "failed" if change.kind == "regression" else "passed"
            parts.append(
                f"<tr><td><code>{_esc(change.case_id)}</code></td>"
                f"<td class='{cls}'>{change.kind}</td><td>{_esc(change.reason)}</td></tr>"
            )
        parts.append(
            f"<tr><td colspan='3'>{len(diff.new_cases)} new, {len(diff.removed_cases)} removed, "
            f"{diff.unchanged} unchanged</td></tr></table></div>"
        )
    parts.append("<h2>Cases</h2><div class='table-wrap'><table>")
    parts.append("<tr><th>Case</th><th>Status</th><th>Scores</th><th>Latency</th></tr>")
    parts.extend(_case_row(case) for case in run.cases)
    parts.append("</table></div></main></body></html>")
    return "\n".join(parts) + "\n"


def render_junit(run: RunResult, diff: BaselineDiff | None = None) -> str:
    """JUnit XML: one testcase per case, plus one per failed threshold and regression."""
    regressions = {c.case_id: c.reason for c in diff.regressions} if diff else {}
    cases = run.case_map()
    # Score drops on cases that still pass are reported as failures too.
    score_only = [cid for cid in regressions if cid in cases and cases[cid].passed]
    failed_thresholds = [t for t in run.thresholds if not t.ok]
    failures = sum(c.status == "failed" for c in run.cases)
    failures += len(failed_thresholds) + len(score_only)
    suites = ET.Element("testsuites", name="llm-eval")
    suite = ET.SubElement(
        suites,
        "testsuite",
        name=run.suite,
        tests=str(len(run.cases) + len(run.thresholds)),
        failures=str(failures),
        errors=str(run.summary.errors),
        time=f"{run.duration_s:.3f}",
        timestamp=run.started_at,
    )
    props = ET.SubElement(suite, "properties")
    for key, value in (("provider", run.provider), ("model", run.model)):
        ET.SubElement(props, "property", name=key, value=value)
    for case in run.cases:
        seconds = f"{case.latency_ms / 1000:.3f}"
        tc = ET.SubElement(suite, "testcase", classname=run.suite, name=case.id, time=seconds)
        if case.status == "error":
            error = _xml_safe(case.error or "error")
            ET.SubElement(tc, "error", message=error).text = error
        elif case.status == "failed":
            detail = "\n".join(f"{s.name}: {s.detail}" for s in case.scores if not s.passed)
            message = ", ".join(s.name for s in case.scores if not s.passed)
            failure = ET.SubElement(tc, "failure", message=_xml_safe(f"failed: {message}"))
            failure.text = _xml_safe(detail)
        elif case.id in regressions:
            failure = ET.SubElement(tc, "failure", message="regression vs baseline")
            failure.text = _xml_safe(regressions[case.id])
        ET.SubElement(tc, "system-out").text = _xml_safe(case.output)
    for check in run.thresholds:
        tc = ET.SubElement(suite, "testcase", classname=f"{run.suite}.thresholds", name=check.name)
        if not check.ok:
            ET.SubElement(
                tc,
                "failure",
                message=f"{check.actual:.3f} is below the required {check.required:.3f}",
            )
    ET.indent(suites)
    return ET.tostring(suites, encoding="unicode", xml_declaration=True) + "\n"


def write_report(path: str | Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
