"""
Eval results reporter. Prints terminal tables from a pytest-json-report JSON file.

Usage:
  python -m tests.eval.report_plugin [eval_*.json]
  (defaults to most recent eval_*.json in cwd)
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_node(nodeid: str, outcome: str, call: dict | None) -> dict:
    parts = nodeid.split("::")
    module = Path(parts[0]).stem.replace("test_", "")
    rest = "::".join(parts[1:])

    if "[" in rest:
        func = rest[: rest.index("[")]
        params_str = rest[rest.index("[") + 1 : rest.rindex("]")]
    else:
        func = rest
        params_str = ""

    model_m = re.search(r"model=([^,\]\-][^\-\]]*)", params_str)
    temp_m = re.search(r"temp=([^,\]]+)", params_str)

    return {
        "module": module,
        "func": func,
        "model": model_m.group(1).strip() if model_m else "—",
        "temp": temp_m.group(1).strip() if temp_m else "—",
        "outcome": outcome,
        "longrepr": (call or {}).get("longrepr", ""),
    }


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

def _table(headers: list[str], rows: list[list[str]], right_cols: set[int] | None = None) -> str:
    right_cols = right_cols or set()
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt_row(cells: list[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            w = widths[i]
            parts.append(str(cell).rjust(w) if i in right_cols else str(cell).ljust(w))
        return "  " + "   ".join(parts)

    divider = "  " + "   ".join("─" * w for w in widths)

    lines = [fmt_row(headers), divider]
    for row in rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def _rate(passed: int, total: int) -> str:
    if total == 0:
        return "—"
    return f"{round(passed / total * 100)}%"


def _bar(passed: int, total: int, width: int = 10) -> str:
    if total == 0:
        return "░" * width
    filled = round(passed / total * width)
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _section(title: str) -> str:
    return f"\n{title}\n{'─' * len(title)}"


def _overall(tests: list[dict]) -> str:
    total = len(tests)
    passed = sum(1 for t in tests if t["outcome"] == "passed")
    failed = sum(1 for t in tests if t["outcome"] == "failed")
    skipped = total - passed - failed
    duration = sum((t.get("call") or {}).get("duration", 0) for t in tests)
    return (
        f"  {passed} passed  {failed} failed  {skipped} skipped  "
        f"({total} total)  {duration:.0f}s"
    )


def _by_model(parsed: list[dict]) -> str:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for p in parsed:
        counts[p["model"]][1] += 1
        if p["outcome"] == "passed":
            counts[p["model"]][0] += 1

    rows = sorted(counts.items(), key=lambda x: -(x[1][0] / x[1][1]) if x[1][1] else 0)
    return _table(
        ["Model", "Pass", "Fail", "Rate", ""],
        [[m, str(pa), str(tot - pa), _rate(pa, tot), _bar(pa, tot)]
         for m, (pa, tot) in rows],
        right_cols={1, 2},
    )


def _by_agent(parsed: list[dict]) -> str:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for p in parsed:
        counts[p["module"]][1] += 1
        if p["outcome"] == "passed":
            counts[p["module"]][0] += 1

    rows = sorted(counts.items(), key=lambda x: -(x[1][0] / x[1][1]) if x[1][1] else 0)
    return _table(
        ["Agent", "Pass", "Fail", "Rate", ""],
        [[a, str(pa), str(tot - pa), _rate(pa, tot), _bar(pa, tot)]
         for a, (pa, tot) in rows],
        right_cols={1, 2},
    )


def _by_test(parsed: list[dict]) -> str:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for p in parsed:
        key = f"{p['module']}::{p['func']}"
        counts[key][1] += 1
        if p["outcome"] == "passed":
            counts[key][0] += 1

    rows = sorted(counts.items(), key=lambda x: -(x[1][0] / x[1][1]) if x[1][1] else 0)
    return _table(
        ["Test", "Pass", "Fail", "Rate"],
        [[t, str(pa), str(tot - pa), _rate(pa, tot)] for t, (pa, tot) in rows],
        right_cols={1, 2},
    )


def _model_agent_crosstab(parsed: list[dict]) -> str:
    agents = sorted({p["module"] for p in parsed})
    counts: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for p in parsed:
        counts[p["model"]][p["module"]][1] += 1
        if p["outcome"] == "passed":
            counts[p["model"]][p["module"]][0] += 1

    def model_avg(m: str) -> float:
        vals = [counts[m][a][0] / counts[m][a][1] for a in agents if counts[m][a][1]]
        return sum(vals) / len(vals) if vals else 0

    models = sorted(counts.keys(), key=lambda m: -model_avg(m))
    rows = []
    for m in models:
        row = [m] + [
            _rate(counts[m][a][0], counts[m][a][1]) if counts[m][a][1] else "—"
            for a in agents
        ]
        rows.append(row)

    return _table(["Model"] + agents, rows)


def _failures(parsed: list[dict]) -> str:
    failed = [p for p in parsed if p["outcome"] == "failed"]
    if not failed:
        return "  (none)"

    lines = []
    for p in failed:
        tag = f"[{p['module']}] {p['func']}  model={p['model']}"
        if p["temp"] != "—":
            tag += f"  temp={p['temp']}"
        lines.append(f"  ✗ {tag}")
        for line in p["longrepr"].splitlines():
            line = line.strip().lstrip("E").strip()
            if line.startswith("assert") or line.startswith("AssertionError") or line.startswith("Score:"):
                lines.append(f"      {line[:120]}")
                break
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_report(path: str) -> str:
    data = json.loads(Path(path).read_text())
    tests = data.get("tests", [])
    if not tests:
        return "No eval tests found in report."

    parsed = [_parse_node(t["nodeid"], t["outcome"], t.get("call")) for t in tests]

    sections = []
    if data.get("partial"):
        sections.append("⚠  PARTIAL RESULTS — run did not complete\n")
    sections += [
        f"Eval report: {path}",
        "",
        _section("OVERALL"),
        _overall(tests),
        _section("BY MODEL"),
        _by_model(parsed),
        _section("BY AGENT"),
        _by_agent(parsed),
        _section("MODEL × AGENT  (pass rate)"),
        _model_agent_crosstab(parsed),
        _section("BY TEST FUNCTION"),
        _by_test(parsed),
        _section("FAILURES"),
        _failures(parsed),
        "",
    ]
    return "\n".join(sections)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        candidates = sorted(Path(".").glob("eval_*.json"), reverse=True)
        if not candidates:
            print("No eval_*.json found. Pass a file path as argument.")
            sys.exit(1)
        path = str(candidates[0])
        print(f"Using most recent: {path}\n")
    print(build_report(path))
