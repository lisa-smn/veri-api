"""
Quality-Check-Script für Factuality Judge + Coherence Tests.

Verifiziert:
1. AUROC/Confidence korrekt (nur bei confidence, sonst N/A)
2. Parsing-Stats vorhanden und konsistent
3. Bootstrap Edge Cases (AUROC skipped resamples)
4. Coherence Skip-Reason konsistent (Test + Doku)
5. Status-Pack Note vorhanden

Usage:
    python scripts/verify_quality_factuality_coherence.py --judge_run <path>
    python scripts/verify_quality_factuality_coherence.py  # auto-detect latest run
"""

import argparse
import json
from pathlib import Path
import re

# ---------------------------
# Check 1: AUROC/Confidence
# ---------------------------


def check_auroc_confidence(judge_run_path: Path) -> tuple[bool, list[str]]:
    """
    Prüft, ob AUROC korrekt basierend auf confidence berechnet wird.

    Returns:
        (pass, messages)
    """
    messages = []
    passed = True

    predictions_path = judge_run_path / "predictions.jsonl"
    summary_md_path = judge_run_path / "summary.md"
    summary_json_path = judge_run_path / "summary.json"

    if not predictions_path.exists():
        return False, [f"❌ predictions.jsonl nicht gefunden: {predictions_path}"]

    if not summary_md_path.exists():
        return False, [f"❌ summary.md nicht gefunden: {summary_md_path}"]

    # Lade predictions
    confidences = []
    has_confidence = False
    with predictions_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                conf = data.get("judge_confidence")
                if conf is not None:
                    has_confidence = True
                    confidences.append(float(conf))
            except (json.JSONDecodeError, ValueError):
                continue

    # Lade summary.md
    with summary_md_path.open("r", encoding="utf-8") as f:
        summary_md = f.read()

    # Prüfe AUROC in summary.md
    auroc_match = re.search(r"\*\*AUROC:\*\*\s*(.+?)(?:\n|$)", summary_md, re.IGNORECASE)
    auroc_line = auroc_match.group(1).strip() if auroc_match else None

    # Prüfe summary.json für auroc_available
    auroc_available = None
    auroc_value = None
    if summary_json_path.exists():
        with summary_json_path.open("r", encoding="utf-8") as f:
            summary_json = json.load(f)
            auroc_available = summary_json.get("auroc_available")
            auroc_value = summary_json.get("auroc")

    # Validierung
    if has_confidence and confidences:
        # Confidence vorhanden: AUROC sollte berechnet werden
        min_conf = min(confidences)
        max_conf = max(confidences)

        # Prüfe Range
        if min_conf < 0.0 or max_conf > 1.0:
            messages.append(
                f"⚠️  Confidence außerhalb [0,1]: min={min_conf:.3f}, max={max_conf:.3f}"
            )
            # Kein hard fail, da clamp im Code passiert

        # Prüfe AUROC
        if auroc_available is False or (auroc_line and "N/A" in auroc_line):
            messages.append(
                f"❌ AUROC ist N/A, obwohl confidence vorhanden ist ({len(confidences)} Beispiele)"
            )
            passed = False
        elif auroc_available is True and auroc_value is not None:
            if (
                "computed on confidence" not in summary_md.lower()
                and "confidence" not in auroc_line.lower()
            ):
                messages.append(
                    "⚠️  AUROC berechnet, aber kein Hinweis auf 'computed on confidence' in summary.md"
                )
            else:
                messages.append(
                    f"✅ AUROC korrekt berechnet auf confidence (n={len(confidences)}, AUROC={auroc_value:.4f})"
                )
        else:
            messages.append(
                f"⚠️  AUROC-Status unklar: available={auroc_available}, value={auroc_value}"
            )
    else:
        # Keine confidence: AUROC sollte N/A sein
        if auroc_available is True or (
            auroc_line and "N/A" not in auroc_line and auroc_value is not None
        ):
            messages.append("❌ AUROC ist berechnet, obwohl keine confidence vorhanden ist")
            passed = False
        else:
            messages.append("✅ AUROC korrekt als N/A markiert (keine confidence vorhanden)")

    return passed, messages


# ---------------------------
# Check 2: Parsing-Stats
# ---------------------------


def check_parsing_stats(judge_run_path: Path) -> tuple[bool, list[str]]:
    """
    Prüft, ob Parsing-Stats vorhanden und konsistent sind.

    Returns:
        (pass, messages)
    """
    messages = []
    passed = True

    summary_md_path = judge_run_path / "summary.md"
    summary_json_path = judge_run_path / "summary.json"
    predictions_path = judge_run_path / "predictions.jsonl"

    if not summary_md_path.exists():
        return False, [f"❌ summary.md nicht gefunden: {summary_md_path}"]

    # Lade summary.md
    with summary_md_path.open("r", encoding="utf-8") as f:
        summary_md = f.read()

    # Extrahiere Parse-Stats aus summary.md
    # Flexible Patterns (mit/ohne **, mit/ohne Klammern)
    # Pattern: "**Parsed JSON (OK):** 5" oder "Parsed JSON (OK): 5"
    json_ok_match = re.search(r"\*\*Parsed JSON\s*\(OK\):\*\*\s*(\d+)", summary_md, re.IGNORECASE)
    if not json_ok_match:
        json_ok_match = re.search(r"Parsed JSON\s*\(OK\):\s*(\d+)", summary_md, re.IGNORECASE)
    regex_fallback_match = re.search(
        r"\*\*Parsed Regex\s*\(Fallback\):\*\*\s*(\d+)", summary_md, re.IGNORECASE
    )
    if not regex_fallback_match:
        regex_fallback_match = re.search(
            r"Parsed Regex\s*\(Fallback\):\s*(\d+)", summary_md, re.IGNORECASE
        )
    parse_failed_match = re.search(r"\*\*Parse Failed:\*\*\s*(\d+)", summary_md, re.IGNORECASE)
    if not parse_failed_match:
        parse_failed_match = re.search(r"Parse Failed:\s*(\d+)", summary_md, re.IGNORECASE)

    json_ok = int(json_ok_match.group(1)) if json_ok_match else None
    regex_fallback = int(regex_fallback_match.group(1)) if regex_fallback_match else None
    parse_failed = int(parse_failed_match.group(1)) if parse_failed_match else None

    # Prüfe, ob Stats vorhanden sind
    if json_ok is None and regex_fallback is None and parse_failed is None:
        messages.append("❌ Parse-Stats nicht in summary.md gefunden")
        passed = False
        return passed, messages

    # Lade n_used aus summary.json
    n_used = None
    if summary_json_path.exists():
        with summary_json_path.open("r", encoding="utf-8") as f:
            summary_json = json.load(f)
            n_used = summary_json.get("n_used")

    # Zähle predictions
    n_predictions = 0
    if predictions_path.exists():
        with predictions_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n_predictions += 1

    # Validierung
    total_parsed = (json_ok or 0) + (regex_fallback or 0)
    total_stats = total_parsed + (parse_failed or 0)

    if n_used is not None:
        # Prüfe Konsistenz mit n_used
        if total_stats != n_used:
            messages.append(f"⚠️  Parse-Stats Summe ({total_stats}) != n_used ({n_used})")
        else:
            messages.append(f"✅ Parse-Stats konsistent mit n_used ({n_used})")

    # Prüfe Anteile
    if total_stats > 0:
        json_pct = 100.0 * (json_ok or 0) / total_stats
        regex_pct = 100.0 * (regex_fallback or 0) / total_stats
        failed_pct = 100.0 * (parse_failed or 0) / total_stats

        messages.append(f"  - JSON OK: {json_ok} ({json_pct:.1f}%)")
        messages.append(f"  - Regex Fallback: {regex_fallback} ({regex_pct:.1f}%)")
        messages.append(f"  - Failed: {parse_failed} ({failed_pct:.1f}%)")

        if parse_failed and parse_failed > 0:
            messages.append(f"⚠️  Parse-Fehler vorhanden ({parse_failed}, {failed_pct:.1f}%)")
            # Warnung, kein hard fail

        if regex_pct > 50:
            messages.append(
                f"⚠️  Hoher Regex-Fallback-Anteil ({regex_pct:.1f}%) - JSON-Parsing sollte primär sein"
            )

    # Prüfe, ob "JSON parsing is primary" Note vorhanden
    if "JSON parsing is primary" not in summary_md and "primary" not in summary_md.lower():
        messages.append("⚠️  Kein Hinweis auf 'JSON parsing is primary' in summary.md")

    return passed, messages


# ---------------------------
# Check 3: Bootstrap Edge Cases
# ---------------------------


def check_bootstrap_edge_cases(judge_run_path: Path) -> tuple[bool, list[str]]:
    """
    Prüft Bootstrap Edge Cases für AUROC.

    Returns:
        (pass, messages)
    """
    messages = []
    passed = True

    summary_md_path = judge_run_path / "summary.md"
    summary_json_path = judge_run_path / "summary.json"

    if not summary_md_path.exists():
        return False, [f"❌ summary.md nicht gefunden: {summary_md_path}"]

    # Lade summary.md
    with summary_md_path.open("r", encoding="utf-8") as f:
        summary_md = f.read()

    # Prüfe AUROC-Status
    auroc_available = None
    auroc_bootstrap_skipped = None
    if summary_json_path.exists():
        with summary_json_path.open("r", encoding="utf-8") as f:
            summary_json = json.load(f)
            auroc_available = summary_json.get("auroc_available")
            auroc_bootstrap_skipped = summary_json.get("auroc_bootstrap_skipped_resamples")

    if auroc_available:
        # AUROC aktiv: prüfe skipped resamples
        if auroc_bootstrap_skipped is not None:
            if auroc_bootstrap_skipped > 0:
                messages.append(
                    f"⚠️  AUROC Bootstrap skipped {auroc_bootstrap_skipped} single-class resamples"
                )
                # Warnung, kein hard fail
            else:
                messages.append("✅ AUROC Bootstrap: keine skipped resamples")
        else:
            # Optional: skipped resamples nicht vorhanden ist ok (wenn nicht gebootstrapped)
            messages.append("ℹ️  AUROC Bootstrap skipped resamples nicht in summary.json (optional)")

        # Prüfe, ob in summary.md erwähnt
        if auroc_bootstrap_skipped and auroc_bootstrap_skipped > 0:
            if "skipped" not in summary_md.lower() or "single-class" not in summary_md.lower():
                messages.append("⚠️  Skipped resamples nicht in summary.md dokumentiert")
    else:
        # AUROC N/A: skipped resamples sollte auch N/A sein
        if auroc_bootstrap_skipped is not None and auroc_bootstrap_skipped != 0:
            messages.append("⚠️  AUROC ist N/A, aber skipped resamples ist nicht None/0")
        else:
            messages.append("✅ AUROC N/A: skipped resamples korrekt als None/0")

    return passed, messages


# ---------------------------
# Check 4: Coherence Skip-Reason
# ---------------------------


def check_coherence_skip_reason() -> tuple[bool, list[str]]:
    """
    Prüft, ob Coherence Skip-Reason konsistent ist (Test + Doku).

    Returns:
        (pass, messages)
    """
    messages = []
    passed = True

    # Lade Test-Datei
    test_path = (
        Path(__file__).parent.parent
        / "tests"
        / "coherence"
        / "test_eval_sumeval_coherence_integration.py"
    )
    if not test_path.exists():
        return False, [f"❌ Test-Datei nicht gefunden: {test_path}"]

    with test_path.open("r", encoding="utf-8") as f:
        test_content = f.read()

    # Extrahiere Skip-Reason aus Test
    # Suche nach pytest.skip(reason="...") mit flexibler Quote-Behandlung
    # Pattern: pytest.skip(reason="..." oder reason='...')
    skip_reason_match = re.search(r'pytest\.skip\(reason=(["\'])(.+?)\1', test_content)
    if not skip_reason_match:
        # Versuche skipif
        skip_reason_match = re.search(r'skipif.*reason=(["\'])(.+?)\1', test_content)
    if not skip_reason_match:
        # Versuche Variable: skip_reason = "..." (kann über mehrere Zeilen gehen)
        skip_reason_match = re.search(
            r'skip_reason\s*=\s*(["\'])(.+?)\1', test_content, re.MULTILINE
        )
    if not skip_reason_match:
        # Versuche direkten String in pytest.skip() ohne reason= (seltener Fall)
        skip_reason_match = re.search(r'pytest\.skip\((["\'])(.+?)\1', test_content)

    if not skip_reason_match:
        messages.append("❌ Skip-Reason nicht im Test gefunden")
        messages.append("   Debug: Suche nach 'skip_reason' oder 'pytest.skip' in Test-Datei")
        passed = False
        return passed, messages

    skip_reason = skip_reason_match.group(2).strip()  # group(2) weil group(1) die Quote ist
    messages.append(f"  Gefundener Skip-Reason im Test: '{skip_reason}'")

    # Prüfe in Doku
    audit_path = (
        Path(__file__).parent.parent / "docs" / "status" / "audit_factuality_coherence_todo.md"
    )
    readme_path = Path(__file__).parent.parent / "README.md"

    found_in_docs = False
    for doc_path in [audit_path, readme_path]:
        if doc_path.exists():
            with doc_path.open("r", encoding="utf-8") as f:
                doc_content = f.read()
                if skip_reason in doc_content:
                    messages.append(f"✅ Skip-Reason gefunden in: {doc_path.name}")
                    found_in_docs = True
                    break

    if not found_in_docs:
        messages.append(f"❌ Skip-Reason '{skip_reason}' nicht in Doku gefunden")
        passed = False

    return passed, messages


# ---------------------------
# Check 5: Status-Pack Note
# ---------------------------


def check_status_pack_note() -> tuple[bool, list[str]]:
    """
    Prüft, ob Status-Pack Note vorhanden ist.

    Returns:
        (pass, messages)
    """
    messages = []
    passed = True

    exec_summary_path = (
        Path(__file__).parent.parent
        / "docs"
        / "status_pack"
        / "2026-01-08"
        / "00_executive_summary.md"
    )
    readme_path = Path(__file__).parent.parent / "docs" / "status_pack" / "2026-01-08" / "README.md"

    # Prüfe Executive Summary
    if exec_summary_path.exists():
        with exec_summary_path.open("r", encoding="utf-8") as f:
            content = f.read()

        has_date_note = "2026-01-08" in content and "2026-01-16" in content
        has_update_note = (
            "Spätere Updates" in content or "spätere Updates" in content or "Updates" in content
        )

        if has_date_note and has_update_note:
            messages.append(f"✅ Status-Pack Note gefunden in: {exec_summary_path.name}")
        else:
            messages.append(
                f"❌ Status-Pack Note fehlt oder unvollständig in: {exec_summary_path.name}"
            )
            passed = False
    else:
        messages.append(f"❌ Executive Summary nicht gefunden: {exec_summary_path}")
        passed = False

    # Optional: Prüfe README
    if readme_path.exists():
        with readme_path.open("r", encoding="utf-8") as f:
            content = f.read()
        if "2026-01-16" in content:
            messages.append("✅ Status-Pack Note auch in README.md gefunden")
        else:
            messages.append("ℹ️  Status-Pack Note nicht in README.md (optional)")

    return passed, messages


# ---------------------------
# Main
# ---------------------------


def find_latest_judge_run() -> Path | None:
    """Findet den neuesten Judge-Run."""
    factuality_dir = Path(__file__).parent.parent / "results" / "evaluation" / "factuality"
    if not factuality_dir.exists():
        return None

    judge_runs = list(factuality_dir.glob("judge_factuality_*"))
    if not judge_runs:
        return None

    # Sortiere nach Modifikationszeit
    judge_runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return judge_runs[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Quality-Check für Factuality Judge + Coherence Tests")
    ap.add_argument(
        "--judge_run", type=str, help="Pfad zu Judge-Run-Verzeichnis (default: auto-detect latest)"
    )
    ap.add_argument(
        "--use_fixture",
        action="store_true",
        help="Verwende Mini-Fixture für Checks 1-3 (tests/fixtures/factuality_judge_run_mini)",
    )
    ap.add_argument("--json_output", type=str, help="Optional: JSON-Report-Pfad")

    args = ap.parse_args()

    # Finde Judge-Run
    judge_run_path = None
    judge_run_source = None

    if args.use_fixture:
        # Fixture-Mode
        fixture_path = (
            Path(__file__).parent.parent / "tests" / "fixtures" / "factuality_judge_run_mini"
        )
        if fixture_path.exists():
            judge_run_path = fixture_path
            judge_run_source = "fixture"
        else:
            print(f"❌ Fixture nicht gefunden: {fixture_path}")
            print(
                "   Erstelle tests/fixtures/factuality_judge_run_mini/ mit predictions.jsonl, summary.json, summary.md"
            )
            exit(1)
    elif args.judge_run:
        # Explizit angegeben
        judge_run_path = Path(args.judge_run)
        judge_run_source = "explicit"
    else:
        # Auto-detect
        judge_run_path = find_latest_judge_run()
        judge_run_source = "auto_detect"

    results = {}
    all_passed = True
    has_not_run = False

    # Checks 1-3 benötigen Judge-Run
    if judge_run_path and judge_run_path.exists():
        source_msg = "Fixture" if judge_run_source == "fixture" else str(judge_run_path)
        print(f"🔍 Quality-Check für: {source_msg}")
        print("=" * 60)

        # Check 1: AUROC/Confidence
        print("\n1️⃣  AUROC/Confidence Check")
        print("-" * 60)
        passed, messages = check_auroc_confidence(judge_run_path)
        results["auroc_confidence"] = {
            "status": "pass" if passed else "fail",
            "passed": passed,
            "messages": messages,
            "reason": None,
        }
        for msg in messages:
            print(f"  {msg}")
        if not passed:
            all_passed = False

        # Check 2: Parsing-Stats
        print("\n2️⃣  Parsing-Stats Check")
        print("-" * 60)
        passed, messages = check_parsing_stats(judge_run_path)
        results["parsing_stats"] = {
            "status": "pass" if passed else "fail",
            "passed": passed,
            "messages": messages,
            "reason": None,
        }
        for msg in messages:
            print(f"  {msg}")
        if not passed:
            all_passed = False

        # Check 3: Bootstrap Edge Cases
        print("\n3️⃣  Bootstrap Edge Cases Check")
        print("-" * 60)
        passed, messages = check_bootstrap_edge_cases(judge_run_path)
        results["bootstrap_edge_cases"] = {
            "status": "pass" if passed else "fail",
            "passed": passed,
            "messages": messages,
            "reason": None,
        }
        for msg in messages:
            print(f"  {msg}")
        if not passed:
            all_passed = False
    else:
        # Kein Judge-Run verfügbar
        has_not_run = True
        reason = "No judge run artifacts found"

        if args.judge_run:
            # Explizit angegeben, aber nicht gefunden -> FAIL
            print(f"❌ Judge-Run nicht gefunden: {args.judge_run}")
            all_passed = False
            reason = f"Judge run path specified but not found: {args.judge_run}"
        else:
            # Auto-detect leer -> NOT RUN (kein Fehler)
            print("ℹ️  Kein Judge-Run gefunden (auto-detect).")
            print("   Checks 1-3 werden nicht ausgeführt (benötigen Judge-Run-Artefakte).")
            print("\n   Nächster Schritt:")
            print("   Um Checks 1-3 auszuführen, führe einen Judge-Run aus:")
            print("   python scripts/eval_frank_factuality_llm_judge.py \\")
            print("     --data data/frank/frank_subset_manifest.jsonl \\")
            print("     --max_examples 50 \\")
            print("     --seed 42 \\")
            print("     --bootstrap_n 200 \\")
            print("     --cache_mode off \\")
            print("     --prompt_version v2_binary \\")
            print("     --judge_n 3 \\")
            print("     --judge_temperature 0.0 \\")
            print("     --judge_aggregation majority")
            print("\n   Oder verwende --use_fixture für Tests mit Mini-Fixture:")
            print("   python scripts/verify_quality_factuality_coherence.py --use_fixture")
            print()

        results["auroc_confidence"] = {
            "status": "not_run",
            "passed": None,
            "messages": [],
            "reason": reason,
        }
        results["parsing_stats"] = {
            "status": "not_run",
            "passed": None,
            "messages": [],
            "reason": reason,
        }
        results["bootstrap_edge_cases"] = {
            "status": "not_run",
            "passed": None,
            "messages": [],
            "reason": reason,
        }

    # Check 4: Coherence Skip-Reason
    print("\n4️⃣  Coherence Skip-Reason Check")
    print("-" * 60)
    passed, messages = check_coherence_skip_reason()
    results["coherence_skip_reason"] = {
        "status": "pass" if passed else "fail",
        "passed": passed,
        "messages": messages,
        "reason": None,
    }
    for msg in messages:
        print(f"  {msg}")
    if not passed:
        all_passed = False

    # Check 5: Status-Pack Note
    print("\n5️⃣  Status-Pack Note Check")
    print("-" * 60)
    passed, messages = check_status_pack_note()
    results["status_pack_note"] = {
        "status": "pass" if passed else "fail",
        "passed": passed,
        "messages": messages,
        "reason": None,
    }
    for msg in messages:
        print(f"  {msg}")
    if not passed:
        all_passed = False

    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)

    for check_name, check_result in results.items():
        status_str = check_result.get("status", "unknown")
        if status_str == "not_run":
            status = "⚠️  NOT RUN"
        elif status_str == "pass":
            status = "✅ PASS"
        elif status_str == "fail":
            status = "❌ FAIL"
        else:
            status = "❓ UNKNOWN"
        print(f"  {check_name}: {status}")
        if status_str == "not_run" and check_result.get("reason"):
            print(f"    Grund: {check_result['reason']}")

    # Exit Code Logik:
    # - 0 wenn alle RUN-Checks PASS sind (NOT RUN ist ok, wenn nicht explizit angefordert)
    # - 1 wenn ein RUN-Check FAIL ist oder --judge_run explizit angegeben aber nicht gefunden
    if all_passed:
        if has_not_run and not args.judge_run:
            print("\n✅ Alle ausführbaren Checks bestanden!")
            print("   (Checks 1-3 nicht ausgeführt: keine Judge-Run-Artefakte gefunden)")
        else:
            print("\n✅ Alle Checks bestanden!")
        exit_code = 0
    else:
        print("\n❌ Einige Checks fehlgeschlagen!")
        exit_code = 1

    # JSON Output
    if args.json_output:
        json_path = Path(args.json_output)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "judge_run": str(judge_run_path) if judge_run_path else None,
                    "judge_run_source": judge_run_source,
                    "all_passed": all_passed,
                    "has_not_run": has_not_run,
                    "results": results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\n📄 JSON-Report gespeichert: {json_path}")

    exit(exit_code)


if __name__ == "__main__":
    main()
