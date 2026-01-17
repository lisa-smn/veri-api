"""
Generiert das Project Status Pack für die Betreuerin.

Sammelt automatisch:
- Evaluation-Ergebnisse aus results/evaluation/**/summary.json
- Repo-Struktur und Dead-Code-Kandidaten
- Explainability-Modul-Informationen
"""

from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def find_summary_jsons(base_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Findet alle summary.json Dateien und lädt sie."""
    summaries = []
    for summary_path in base_dir.rglob("summary.json"):
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                summaries.append((summary_path, data))
        except Exception as e:
            print(f"Warnung: Konnte {summary_path} nicht laden: {e}")
    return summaries


def classify_run(summary_path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    """Klassifiziert einen Run (Agent/Baseline/Judge/Stress)."""
    run_dir = summary_path.parent
    run_id = run_dir.name

    # Method aus summary
    method = summary.get("method", "unknown")
    baseline_type = summary.get("baseline_type")

    # Aus Pfad ableiten
    if "coherence_judge" in str(summary_path):
        method = "llm_judge"
    elif "coherence_baselines" in str(summary_path) or "factuality_baselines" in str(summary_path):
        method = baseline_type or "baseline"
    elif "coherence_stress" in str(summary_path):
        method = "stress"
    elif (
        "coherence" in str(summary_path)
        and "baselines" not in str(summary_path)
        and "judge" not in str(summary_path)
    ) or ("factuality" in str(summary_path) and "baselines" not in str(summary_path)):
        method = "agent"

    # Metadata laden
    metadata_path = run_dir / "run_metadata.json"
    metadata = {}
    if metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            pass

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "method": method,
        "summary": summary,
        "metadata": metadata,
    }


def extract_coherence_results(
    summaries: list[tuple[Path, dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Extrahiert Coherence-Ergebnisse."""
    results = defaultdict(list)

    for summary_path, summary in summaries:
        if "coherence" not in str(summary_path):
            continue

        run_info = classify_run(summary_path, summary)
        method = run_info["method"]

        if method in ["agent", "llm_judge", "baseline"]:
            results[method].append(run_info)

    return dict(results)


def extract_factuality_results(
    summaries: list[tuple[Path, dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Extrahiert Factuality-Ergebnisse."""
    results = defaultdict(list)

    for summary_path, summary in summaries:
        if "factuality" not in str(summary_path):
            continue

        run_info = classify_run(summary_path, summary)
        method = run_info["method"]

        if method in ["agent", "baseline"]:
            results[method].append(run_info)

    return dict(results)


def scan_repo_for_dead_code() -> list[dict[str, Any]]:
    """Scannt Repo nach Dead-Code-Kandidaten."""
    candidates = []

    # Scripts mit "archive" im Namen
    scripts_dir = ROOT / "scripts"
    for script_file in scripts_dir.glob("*.py"):
        if "archive" in script_file.name.lower():
            candidates.append(
                {
                    "path": str(script_file.relative_to(ROOT)),
                    "type": "script",
                    "status": "ARCHIVE",
                    "reason": "Dateiname enthält 'archive'",
                    "risk": "low",
                }
            )

    # Doppelte Eval-Scripts (v1/v2)
    eval_scripts = list(scripts_dir.glob("eval_*.py"))
    eval_names = {}
    for script in eval_scripts:
        base = re.sub(r"_v\d+$", "", script.stem)
        if base in eval_names:
            candidates.append(
                {
                    "path": str(script.relative_to(ROOT)),
                    "type": "script",
                    "status": "REVIEW",
                    "reason": f"Mögliche Duplikate: {eval_names[base]} vs {script.name}",
                    "risk": "medium",
                }
            )
        eval_names[base] = script.name

    # Ablation-Module
    ablation_files = list(
        (ROOT / "app" / "services" / "agents" / "factuality").glob("*ablation*.py")
    )
    for f in ablation_files:
        candidates.append(
            {
                "path": str(f.relative_to(ROOT)),
                "type": "module",
                "status": "REVIEW",
                "reason": "Ablation-Modul (möglicherweise experimentell)",
                "risk": "medium",
            }
        )

    return candidates


def main():
    """Hauptfunktion: Generiert Status-Pack-Daten."""
    today = datetime.now().strftime("%Y-%m-%d")
    status_dir = ROOT / "docs" / "status_pack" / today
    status_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generiere Status-Pack für {today}...")

    # Sammle Evaluation-Ergebnisse
    eval_dir = ROOT / "results" / "evaluation"
    summaries = find_summary_jsons(eval_dir)

    coherence_results = extract_coherence_results(summaries)
    factuality_results = extract_factuality_results(summaries)

    # Dead-Code-Scan
    dead_code = scan_repo_for_dead_code()

    # Speichere Daten für Dokumentation
    data_file = status_dir / "status_pack_data.json"
    with data_file.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "coherence": coherence_results,
                "factuality": factuality_results,
                "dead_code": dead_code,
                "total_runs": len(summaries),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Status-Pack-Daten gespeichert: {data_file}")
    print(f"  - Coherence Runs: {sum(len(v) for v in coherence_results.values())}")
    print(f"  - Factuality Runs: {sum(len(v) for v in factuality_results.values())}")
    print(f"  - Dead-Code-Kandidaten: {len(dead_code)}")


if __name__ == "__main__":
    main()
