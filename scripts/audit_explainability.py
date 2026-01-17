#!/usr/bin/env python3
"""
Mini-Audit Script für Explainability-Modul.

Lädt Fixtures, baut ExplainabilityResult, schreibt Report.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.explainability.explainability_service import ExplainabilityService

FIXTURES_DIR = ROOT / "tests" / "fixtures"


def load_fixture(name: str) -> dict:
    """Lädt ein Fixture-File."""
    path = FIXTURES_DIR / f"explainability_input_{name}.json"
    if not path.exists():
        print(f"⚠️  Fixture nicht gefunden: {path}", file=sys.stderr)
        return None
    with path.open() as f:
        return json.load(f)


def check_contract(result) -> tuple[bool, list[str]]:
    """Prüft Contract (required fields, types, ranges)."""
    issues = []

    # Required fields
    if not hasattr(result, "summary"):
        issues.append("Missing: summary")
    if not hasattr(result, "findings"):
        issues.append("Missing: findings")
    if not hasattr(result, "by_dimension"):
        issues.append("Missing: by_dimension")
    if not hasattr(result, "top_spans"):
        issues.append("Missing: top_spans")
    if not hasattr(result, "stats"):
        issues.append("Missing: stats")
    if not hasattr(result, "version"):
        issues.append("Missing: version")

    # Types
    if not isinstance(result.summary, list):
        issues.append("summary must be list")
    if not isinstance(result.findings, list):
        issues.append("findings must be list")
    if not isinstance(result.by_dimension, dict):
        issues.append("by_dimension must be dict")

    # Stats ranges
    if result.stats.coverage_ratio < 0.0 or result.stats.coverage_ratio > 1.0:
        issues.append(f"coverage_ratio out of range: {result.stats.coverage_ratio}")

    # Dimensions present
    from app.services.explainability.explainability_models import Dimension

    for dim in Dimension:
        if dim not in result.by_dimension:
            issues.append(f"Missing dimension: {dim}")

    return len(issues) == 0, issues


def check_determinism(service, input_data, n_runs: int = 5) -> tuple[bool, str]:
    """Prüft Determinismus (n_runs identischer Input → identischer Output)."""
    results = []
    for _ in range(n_runs):
        result = service.build(input_data, input_data["summary_text"])
        results.append(result.model_dump())

    first = results[0]
    for i, result in enumerate(results[1:], 1):
        if result != first:
            return False, f"Run {i + 1} differs from first run"

    return True, f"All {n_runs} runs produced identical output"


def write_report(
    fixture_name: str,
    result,
    contract_pass: bool,
    contract_issues: list[str],
    determinism_pass: bool,
    determinism_msg: str,
    out_path: Path,
) -> None:
    """Schreibt Audit-Report als Markdown."""
    lines = [
        "# Explainability-Modul: Mini-Audit",
        "",
        f"**Datum:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Fixture:** {fixture_name}",
        f"**Version:** {result.version}",
        "",
        "---",
        "",
        "## Contract Check",
        "",
        f"**Status:** {'✅ PASS' if contract_pass else '❌ FAIL'}",
        "",
    ]

    if contract_issues:
        lines.append("**Issues:**")
        for issue in contract_issues:
            lines.append(f"- ⚠️  {issue}")
    else:
        lines.append("✅ Alle Contract-Checks bestanden")

    lines.extend(
        [
            "",
            "## Determinism Check",
            "",
            f"**Status:** {'✅ PASS' if determinism_pass else '❌ FAIL'}",
            f"**Message:** {determinism_msg}",
            "",
            "---",
            "",
            "## Beispiel-Auszug",
            "",
            f"**Anzahl Findings:** {result.stats.num_findings}",
            f"- High: {result.stats.num_high_severity}",
            f"- Medium: {result.stats.num_medium_severity}",
            f"- Low: {result.stats.num_low_severity}",
            "",
            f"**Coverage:** {result.stats.coverage_chars} Zeichen ({result.stats.coverage_ratio:.2%})",
            "",
            "### Findings pro Dimension:",
            "",
        ]
    )

    from app.services.explainability.explainability_models import Dimension

    for dim in Dimension:
        count = len(result.by_dimension[dim])
        lines.append(f"- **{dim.value}:** {count}")

    lines.extend(
        [
            "",
            "### Top 3 Spans:",
            "",
        ]
    )

    for i, top_span in enumerate(result.top_spans[:3], 1):
        text = (top_span.span.text or "")[:70]
        if len(top_span.span.text or "") > 70:
            text += "..."
        lines.append(
            f"{i}. [{top_span.dimension.value}, {top_span.severity}] "
            f"({top_span.span.start_char}-{top_span.span.end_char}): "
            f"„{text}“ (score: {top_span.rank_score:.2f})"
        )

    lines.extend(
        [
            "",
            "### Executive Summary:",
            "",
        ]
    )

    for sentence in result.summary[:3]:
        lines.append(f"- {sentence}")

    lines.extend(
        [
            "",
            "---",
            "",
            "## Persistence Check",
            "",
            "⚠️  Persistence-Checks sind optional und werden nur ausgeführt, wenn DB verfügbar ist.",
            "Siehe `tests/explainability/test_explainability_persistence_*.py` für Details.",
            "",
        ]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ Report geschrieben: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Explainability-Modul Mini-Audit")
    ap.add_argument(
        "--fixture",
        type=str,
        default="minimal",
        choices=["minimal", "mixed", "edgecases"],
        help="Fixture-Name (default: minimal)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="docs/status/explainability_audit.md",
        help="Output-Pfad (default: docs/status/explainability_audit.md)",
    )

    args = ap.parse_args()

    print(f"🔍 Explainability Audit: {args.fixture}")
    print("")

    # Load fixture
    input_data = load_fixture(args.fixture)
    if not input_data:
        print(f"❌ Fixture nicht gefunden: {args.fixture}", file=sys.stderr)
        sys.exit(1)

    # Build Explainability
    service = ExplainabilityService()
    result = service.build(input_data, input_data["summary_text"])

    print(f"✅ ExplainabilityResult gebaut (Version: {result.version})")
    print(f"   Findings: {result.stats.num_findings}")
    print(f"   Top-Spans: {len(result.top_spans)}")
    print("")

    # Contract check
    print("📋 Contract Check...")
    contract_pass, contract_issues = check_contract(result)
    if contract_pass:
        print("  ✅ PASS")
    else:
        print(f"  ❌ FAIL ({len(contract_issues)} issues)")

    # Determinism check
    print("🔄 Determinism Check...")
    determinism_pass, determinism_msg = check_determinism(service, input_data)
    if determinism_pass:
        print(f"  ✅ PASS: {determinism_msg}")
    else:
        print(f"  ❌ FAIL: {determinism_msg}")

    # Write report
    print("")
    print("📝 Report generieren...")
    out_path = Path(args.out)
    write_report(
        args.fixture,
        result,
        contract_pass,
        contract_issues,
        determinism_pass,
        determinism_msg,
        out_path,
    )

    print("")
    print("✅ Audit abgeschlossen!")

    # Exit code
    if not contract_pass or not determinism_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
