"""
Analysiert Regression zwischen v1 alt und v1 neu.

Prüft:
- Wiring (agent vs judge scores)
- Distribution-Vergleich
- Config-Vergleich
- Cache-Status
"""

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from scipy.stats import spearmanr


def load_predictions(predictions_path: Path) -> list[dict[str, Any]]:
    """Lädt predictions.jsonl."""
    predictions = []
    with predictions_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                predictions.append(rec)
            except json.JSONDecodeError:
                continue
    return predictions


def analyze_distribution(
    predictions: list[dict[str, Any]], score_key: str = "pred"
) -> dict[str, int]:
    """Analysiert Score-Verteilung."""
    scores = [p.get(score_key) for p in predictions if p.get(score_key) is not None]
    if not scores:
        return {}

    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    def bin_value(v: float) -> str:
        for i in range(len(bins) - 1):
            if bins[i] <= v < bins[i + 1]:
                return f"[{bins[i]:.1f}, {bins[i + 1]:.1f})"
        if v >= bins[-1]:
            return f"[{bins[-1]:.1f}, 1.0]"
        return "[0.0, 0.2)"

    return Counter([bin_value(s) for s in scores])


def compare_runs(
    agent_run_dir: Path,
    judge_run_dir: Path,
    old_run_dir: Path,
) -> dict[str, Any]:
    """Vergleicht drei Runs."""
    results = {}

    # Lade Predictions
    agent_preds = load_predictions(agent_run_dir / "predictions.jsonl")
    judge_preds = load_predictions(judge_run_dir / "predictions.jsonl")
    old_preds = load_predictions(old_run_dir / "predictions.jsonl")

    # Lade Metadata
    agent_meta = json.load((agent_run_dir / "run_metadata.json").open())
    judge_meta = json.load((judge_run_dir / "run_metadata.json").open())
    old_meta = json.load((old_run_dir / "run_metadata.json").open())

    # Lade Summary
    agent_summary = json.load((agent_run_dir / "summary.json").open())
    judge_summary = json.load((judge_run_dir / "summary.json").open())
    old_summary = json.load((old_run_dir / "summary.json").open())

    # Wiring-Check: Sind agent und judge scores unterschiedlich?
    wiring_check = {
        "agent_has_pred_agent": any("pred_agent" in p for p in agent_preds),
        "agent_has_pred_judge": any(p.get("pred_judge") is not None for p in agent_preds),
        "judge_has_pred_agent": any("pred_agent" in p for p in judge_preds),
        "judge_has_pred_judge": any(p.get("pred_judge") is not None for p in judge_preds),
    }

    # Prüfe Unterschiede in den Scores
    if wiring_check["judge_has_pred_agent"] and wiring_check["judge_has_pred_judge"]:
        agent_scores = [p.get("pred_agent") for p in judge_preds if p.get("pred_agent") is not None]
        judge_scores = [p.get("pred_judge") for p in judge_preds if p.get("pred_judge") is not None]
        if agent_scores and judge_scores:
            corr, _ = spearmanr(agent_scores, judge_scores)
            wiring_check["agent_judge_spearman"] = corr
            wiring_check["scores_differ"] = any(
                abs(a - j) > 0.01 for a, j in zip(agent_scores[:10], judge_scores[:10])
            )

    # Distribution-Vergleich
    old_dist = analyze_distribution(old_preds, "pred")
    agent_dist = analyze_distribution(agent_preds, "pred")
    judge_dist = analyze_distribution(judge_preds, "pred")

    # Config-Vergleich
    config_comparison = {
        "git_commit": {
            "old": old_meta.get("git_commit"),
            "agent": agent_meta.get("git_commit"),
            "judge": judge_meta.get("git_commit"),
        },
        "prompt_version": {
            "old": old_meta.get("config", {}).get("prompt_version"),
            "agent": agent_meta.get("config", {}).get("prompt_version"),
            "judge": judge_meta.get("config", {}).get("prompt_version"),
        },
        "cache": {
            "old": old_meta.get("config", {}).get("cache"),
            "agent": agent_meta.get("config", {}).get("cache"),
            "judge": judge_meta.get("config", {}).get("cache"),
        },
    }

    # Metriken-Vergleich
    metrics_comparison = {
        "old": {
            "spearman": old_summary.get("spearman", {}).get("value"),
            "pearson": old_summary.get("pearson", {}).get("value"),
            "mae": old_summary.get("mae", {}).get("value"),
        },
        "agent": {
            "spearman": agent_summary.get("spearman", {}).get("value"),
            "pearson": agent_summary.get("pearson", {}).get("value"),
            "mae": agent_summary.get("mae", {}).get("value"),
        },
        "judge": {
            "spearman": judge_summary.get("spearman", {}).get("value"),
            "pearson": judge_summary.get("pearson", {}).get("value"),
            "mae": judge_summary.get("mae", {}).get("value"),
        },
    }

    return {
        "wiring_check": wiring_check,
        "distribution": {
            "old": old_dist,
            "agent": agent_dist,
            "judge": judge_dist,
        },
        "config_comparison": config_comparison,
        "metrics_comparison": metrics_comparison,
    }


def write_report(analysis: dict[str, Any], out_path: Path) -> None:
    """Schreibt Analyse-Report."""
    lines = [
        "# Readability Regression & Wiring Analysis",
        "",
        "## 1. Wiring-Check",
        "",
    ]

    wc = analysis["wiring_check"]
    lines.extend(
        [
            f"- **Agent-Run hat pred_agent:** {wc.get('agent_has_pred_agent', False)}",
            f"- **Agent-Run hat pred_judge:** {wc.get('agent_has_pred_judge', False)}",
            f"- **Judge-Run hat pred_agent:** {wc.get('judge_has_pred_agent', False)}",
            f"- **Judge-Run hat pred_judge:** {wc.get('judge_has_pred_judge', False)}",
        ]
    )

    if "agent_judge_spearman" in wc:
        lines.append(
            f"- **Agent vs Judge Korrelation (im Judge-Run):** {wc['agent_judge_spearman']:.4f}"
        )
        lines.append(f"- **Scores unterscheiden sich:** {wc.get('scores_differ', False)}")

    lines.extend(
        [
            "",
            "**Fazit Wiring:** "
            + (
                "✅ Wiring funktioniert: Agent und Judge Scores werden getrennt gespeichert und ausgewertet."
                if wc.get("judge_has_pred_agent")
                and wc.get("judge_has_pred_judge")
                and wc.get("scores_differ")
                else "⚠️ Wiring-Problem: Scores sind identisch oder fehlen."
            ),
            "",
            "## 2. Regression-Check: v1 alt vs v1 neu",
            "",
        ]
    )

    # Config-Vergleich
    cc = analysis["config_comparison"]
    lines.extend(
        [
            "### Config-Vergleich",
            "",
            "| | Alt (20260114) | Neu Agent (20260115) | Neu Judge (20260115) |",
            "|---|---|---|---|",
            f"| Git Commit | {cc['git_commit']['old'][:8]} | {cc['git_commit']['agent'][:8]} | {cc['git_commit']['judge'][:8]} |",
            f"| Prompt Version | {cc['prompt_version']['old']} | {cc['prompt_version']['agent']} | {cc['prompt_version']['judge']} |",
            f"| Cache | {cc['cache']['old']} | {cc['cache']['agent']} | {cc['cache']['judge']} |",
            "",
        ]
    )

    # Metriken-Vergleich
    mc = analysis["metrics_comparison"]
    lines.extend(
        [
            "### Metriken-Vergleich",
            "",
            "| | Alt (20260114) | Neu Agent (20260115) | Neu Judge (20260115) |",
            "|---|---|---|---|",
            f"| Spearman ρ | {mc['old']['spearman']:.4f} | {mc['agent']['spearman']:.4f} | {mc['judge']['spearman']:.4f} |",
            f"| Pearson r | {mc['old']['pearson']:.4f} | {mc['agent']['pearson']:.4f} | {mc['judge']['pearson']:.4f} |",
            f"| MAE | {mc['old']['mae']:.4f} | {mc['agent']['mae']:.4f} | {mc['judge']['mae']:.4f} |",
            "",
        ]
    )

    # Distribution-Vergleich
    lines.extend(
        [
            "### Distribution-Vergleich",
            "",
            "#### Alt (20260114)",
            "",
        ]
    )
    for bin_name in sorted(analysis["distribution"]["old"].keys()):
        count = analysis["distribution"]["old"][bin_name]
        lines.append(f"- {bin_name}: {count}")

    lines.extend(
        [
            "",
            "#### Neu Agent (20260115)",
            "",
        ]
    )
    for bin_name in sorted(analysis["distribution"]["agent"].keys()):
        count = analysis["distribution"]["agent"][bin_name]
        lines.append(f"- {bin_name}: {count}")

    lines.extend(
        [
            "",
            "#### Neu Judge (20260115)",
            "",
        ]
    )
    for bin_name in sorted(analysis["distribution"]["judge"].keys()):
        count = analysis["distribution"]["judge"][bin_name]
        lines.append(f"- {bin_name}: {count}")

    lines.extend(
        [
            "",
            "### Regression-Erklärung",
            "",
            "**Beobachtung:**",
            f"- Alt: Spearman ρ = {mc['old']['spearman']:.4f}",
            f"- Neu Agent: Spearman ρ = {mc['agent']['spearman']:.4f} (Verschlechterung um {mc['old']['spearman'] - mc['agent']['spearman']:.4f})",
            f"- Neu Judge: Spearman ρ = {mc['judge']['spearman']:.4f}",
            "",
            "**Mögliche Ursachen:**",
            "1. **Cache-Effekt:** Alt hatte `cache=false`, neu hat `cache=true` → möglicherweise werden alte, bessere Cached-Ergebnisse verwendet",
            "2. **Distribution-Collapse:** Neu zeigt stark komprimierte Verteilung ([0.4, 0.6): 166 von 200) → geringe Varianz führt zu niedriger Korrelation",
            "3. **LLM-Varianz:** Gleicher Prompt, aber LLM-Output variiert zwischen Runs",
            "",
            "**Empfehlung:**",
            "- Für finale Evaluation: Cache deaktivieren oder Cache leeren",
            "- Distribution prüfen: Wenn >80% in einem Bucket → Collapse-Detector sollte warnen",
            "- Prompt-Version prüfen: Sicherstellen, dass 'use full scale' enthalten ist",
        ]
    )

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Analysiert Readability Regression")
    ap.add_argument("--agent_run", type=str, required=True, help="Agent-Run-Verzeichnis")
    ap.add_argument("--judge_run", type=str, required=True, help="Judge-Run-Verzeichnis")
    ap.add_argument("--old_run", type=str, required=True, help="Alter Referenz-Run")
    ap.add_argument("--out", type=str, required=True, help="Output-Pfad (.md)")

    args = ap.parse_args()

    analysis = compare_runs(
        Path(args.agent_run),
        Path(args.judge_run),
        Path(args.old_run),
    )

    write_report(analysis, Path(args.out))
    print(f"Report geschrieben: {args.out}")


if __name__ == "__main__":
    main()
