#!/usr/bin/env python3
"""
Sammelt kapitelrelevante Evaluation-Runs (Agent v1 vs. v2) für Factuality, Coherence und Readability.

Erstellt eine kanonische Übersicht mit v1/v2-Paaren, Metriken und Delta-Berechnungen.
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
from typing import Any

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@dataclass
class RunInfo:
    """Informationen zu einem einzelnen Run."""
    run_id: str
    run_path: str
    dimension: str
    dataset: str
    n_used: int
    prompt_version: str
    model: str
    seed: int | None
    dataset_signature: str | None
    metrics: dict[str, Any]
    ignored_candidates: list[str] = None  # Andere Runs in derselben Gruppe


def load_json(path: Path) -> dict[str, Any] | None:
    """Lädt JSON-Datei."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def extract_metrics(summary_path: Path, dimension: str) -> dict[str, Any]:
    """Extrahiert Metriken aus summary.json je nach Dimension."""
    data = load_json(summary_path)
    if not data:
        return {}

    metrics = {}

    if dimension == "factuality":
        # Binäre Metriken
        if "metrics" in data:
            m = data["metrics"]
            for key in ["precision", "recall", "f1", "specificity", "balanced_accuracy", "mcc", "auroc", "accuracy"]:
                if key in m:
                    val = m[key]
                    if isinstance(val, dict) and "value" in val:
                        metrics[key] = val["value"]
                    elif isinstance(val, (int, float)):
                        metrics[key] = val

        # Confusion Matrix
        if "counts" in data:
            metrics["confusion_matrix"] = {
                "tp": data["counts"].get("tp", 0),
                "fp": data["counts"].get("fp", 0),
                "tn": data["counts"].get("tn", 0),
                "fn": data["counts"].get("fn", 0),
            }

        # Klassenverteilung
        if "counts" in data:
            total = sum(data["counts"].values())
            if total > 0:
                metrics["class_distribution"] = {
                    "error": data["counts"].get("tp", 0) + data["counts"].get("fn", 0),
                    "no_error": data["counts"].get("tn", 0) + data["counts"].get("fp", 0),
                    "total": total,
                }

    elif dimension in ("coherence", "readability"):
        # Regression-Metriken
        for key in ["pearson", "spearman", "mae", "rmse", "r2", "r_squared"]:
            if key in data:
                val = data[key]
                if isinstance(val, dict) and "value" in val:
                    metrics[key] = val["value"]
                elif isinstance(val, (int, float)):
                    metrics[key] = val

        # Aliases
        if "spearman" in metrics:
            metrics["spearman_rho"] = metrics["spearman"]
        if "pearson" in metrics:
            metrics["pearson_r"] = metrics["pearson"]
        if "r2" in metrics:
            metrics["r_squared"] = metrics["r2"]
        if "r_squared" in metrics:
            metrics["r2"] = metrics["r_squared"]

    return metrics


def infer_dimension_from_path(run_path: Path) -> str | None:
    """Inferiert Dimension aus Pfad."""
    path_str = str(run_path).lower()
    if "factuality" in path_str:
        return "factuality"
    if "coherence" in path_str:
        return "coherence"
    if "readability" in path_str:
        return "readability"
    return None


def infer_dataset_from_metadata(metadata: dict[str, Any], run_path: Path) -> str | None:
    """Inferiert Dataset aus Metadaten oder Pfad."""
    # From metadata
    for key in ["dataset", "data_path", "manifest_path", "manifest", "dataset_path"]:
        if key in metadata and metadata[key]:
            val = str(metadata[key]).lower()
            if "frank" in val:
                return "FRANK"
            if "sumeval" in val:
                return "SummEval"
            if "finesumfact" in val:
                return "FineSumFact"

    # From path
    path_str = str(run_path).lower()
    if "frank" in path_str:
        return "FRANK"
    if "sumeval" in path_str:
        return "SummEval"
    if "finesumfact" in path_str:
        return "FineSumFact"

    # Default für Coherence/Readability: SummEval (wenn Dimension bekannt)
    dimension = infer_dimension_from_path(run_path)
    if dimension in ("coherence", "readability"):
        return "SummEval"

    return None


def normalize_prompt_version(metadata: dict[str, Any], run_path: Path) -> str | None:
    """Extrahiert prompt_version aus Metadaten."""
    # Top level
    if "prompt_version" in metadata and metadata["prompt_version"]:
        return str(metadata["prompt_version"])

    # In config
    config = metadata.get("config", {})
    if "prompt_version" in config and config["prompt_version"]:
        return str(config["prompt_version"])

    # From run_id
    run_id = metadata.get("run_id", run_path.name)
    if "_v" in run_id:
        parts = run_id.split("_v")
        if len(parts) > 1:
            version_part = parts[1].split("_")[0]
            if version_part and version_part[0] == "v":
                return version_part

    return None


def is_chapter_relevant(run_info: RunInfo) -> bool:
    """Prüft, ob Run kapitelrelevant ist."""
    # Dimension muss factuality, coherence oder readability sein
    if run_info.dimension not in ("factuality", "coherence", "readability"):
        return False

    # n_used muss 200 sein
    if run_info.n_used != 200:
        return False

    # prompt_version muss v1 oder v2 sein
    if run_info.prompt_version not in ("v1", "v2"):
        return False

    # Dataset-Checks
    if run_info.dimension == "factuality":
        if run_info.dataset != "FRANK":
            return False
    elif run_info.dimension in ("coherence", "readability"):
        if run_info.dataset != "SummEval":
            return False

    # Keine Judge-Runs, keine Baselines
    run_path_lower = run_info.run_path.lower()
    if "judge" in run_path_lower or "baseline" in run_path_lower:
        return False
    if "rouge" in run_path_lower or "bertscore" in run_path_lower:
        return False
    if "flesch" in run_path_lower or "fog" in run_path_lower:
        return False

    return True


def find_all_runs(runs_root: Path) -> list[RunInfo]:
    """Findet alle Runs rekursiv."""
    runs = []

    # Suche nach run_metadata.json oder summary.json
    for metadata_path in runs_root.rglob("run_metadata.json"):
        run_dir = metadata_path.parent
        summary_path = run_dir / "summary.json"

        if not summary_path.exists():
            continue

        metadata = load_json(metadata_path)
        if not metadata:
            continue

        summary = load_json(summary_path)
        if not summary:
            continue

        # Extrahiere Informationen
        run_id = metadata.get("run_id", run_dir.name)
        dimension = infer_dimension_from_path(run_dir) or metadata.get("dimension") or metadata.get("agent")
        dataset = infer_dataset_from_metadata(metadata, run_dir)
        prompt_version = normalize_prompt_version(metadata, run_dir)
        n_used = metadata.get("n_used") or summary.get("n_used") or summary.get("n_seen")

        # Model
        model = metadata.get("model") or metadata.get("llm_model")
        if not model or model == "unknown":
            config = metadata.get("config", {})
            model = config.get("llm_model") or config.get("model") or model

        # Seed
        seed = metadata.get("seed")
        if seed is None:
            config = metadata.get("config", {})
            seed = config.get("seed")

        # Dataset signature
        dataset_signature = metadata.get("dataset_signature")

        # Metriken
        metrics = extract_metrics(summary_path, dimension or "unknown")

        run_info = RunInfo(
            run_id=run_id,
            run_path=str(run_dir),
            dimension=dimension or "unknown",
            dataset=dataset or "unknown",
            n_used=n_used or 0,
            prompt_version=prompt_version or "unknown",
            model=model or "unknown",
            seed=seed,
            dataset_signature=dataset_signature,
            metrics=metrics,
        )

        if is_chapter_relevant(run_info):
            runs.append(run_info)

    return runs


def group_runs_by_conditions(runs: list[RunInfo]) -> dict[tuple, list[RunInfo]]:
    """Gruppiert Runs nach identischen Bedingungen (außer prompt_version)."""
    groups = defaultdict(list)

    for run in runs:
        # Gruppierungsschlüssel (ohne prompt_version und dataset_signature)
        # dataset_signature kann zwischen Runs variieren, auch wenn sie auf demselben Dataset basieren
        key = (
            run.dimension,
            run.dataset,
            run.model,
            run.seed,
            run.n_used,
        )
        groups[key].append(run)

    return groups


def select_best_pair(group: list[RunInfo]) -> tuple[RunInfo | None, RunInfo | None, list[str]]:
    """Wählt das beste v1/v2-Paar aus einer Gruppe (neueste Runs bevorzugt)."""
    v1_runs = [r for r in group if r.prompt_version == "v1"]
    v2_runs = [r for r in group if r.prompt_version == "v2"]

    # Sortiere nach Run-ID (enthält Timestamp)
    v1_runs.sort(key=lambda r: r.run_id, reverse=True)
    v2_runs.sort(key=lambda r: r.run_id, reverse=True)

    v1_selected = v1_runs[0] if v1_runs else None
    v2_selected = v2_runs[0] if v2_runs else None

    # Ignorierte Kandidaten
    ignored = []
    if len(v1_runs) > 1:
        ignored.extend([r.run_id for r in v1_runs[1:]])
    if len(v2_runs) > 1:
        ignored.extend([r.run_id for r in v2_runs[1:]])

    return v1_selected, v2_selected, ignored


def validate_pair(v1: RunInfo, v2: RunInfo) -> tuple[bool, list[str]]:
    """Validiert, ob v1/v2-Paar konsistent ist."""
    errors = []
    warnings = []

    # Harte Validierungen (Fehler)
    if v1.n_used != v2.n_used:
        errors.append(f"n_used differs: v1={v1.n_used}, v2={v2.n_used}")

    if v1.dataset != v2.dataset:
        errors.append(f"dataset differs: v1={v1.dataset}, v2={v2.dataset}")

    if v1.model != v2.model:
        errors.append(f"model differs: v1={v1.model}, v2={v2.model}")

    if v1.dataset_signature and v2.dataset_signature:
        if v1.dataset_signature != v2.dataset_signature:
            errors.append(f"dataset_signature differs: v1={v1.dataset_signature[:16]}..., v2={v2.dataset_signature[:16]}...")

    # Warnungen
    if v1.seed != v2.seed:
        warnings.append(f"seed differs: v1={v1.seed}, v2={v2.seed}")

    return len(errors) == 0, errors + warnings


def calculate_delta(v1_metrics: dict[str, Any], v2_metrics: dict[str, Any]) -> dict[str, Any]:
    """Berechnet Delta-Metriken (v2 - v1)."""
    delta = {}

    # Alle numerischen Metriken
    all_keys = set(v1_metrics.keys()) | set(v2_metrics.keys())

    for key in all_keys:
        v1_val = v1_metrics.get(key)
        v2_val = v2_metrics.get(key)

        # Nur numerische Werte
        if isinstance(v1_val, (int, float)) and isinstance(v2_val, (int, float)):
            delta[f"delta_{key}"] = v2_val - v1_val

    return delta


def write_json_output(pairs: dict[str, dict[str, Any]], output_path: Path):
    """Schreibt JSON-Output."""
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)


def write_markdown_output(pairs: dict[str, dict[str, Any]], output_path: Path):
    """Schreibt Markdown-Output."""
    lines = [
        "# Kapitel 6: Kanonische Run-Übersicht",
        "",
        "**Erstellt:** Automatisch generiert",
        "",
        "---",
        "",
        "## Zusammenfassung",
        "",
    ]

    # Zähle vollständige Paare
    complete_pairs = sum(1 for p in pairs.values() if p.get("v1") and p.get("v2"))
    incomplete = [dim for dim, p in pairs.items() if not (p.get("v1") and p.get("v2"))]

    lines.append(f"- **Vollständige v1/v2-Paare:** {complete_pairs}/{len(pairs)}")
    if incomplete:
        lines.append(f"- **Unvollständig:** {', '.join(incomplete)}")
    lines.append("")

    # Pro Dimension
    for dimension in ["factuality", "coherence", "readability"]:
        if dimension not in pairs:
            continue

        pair_data = pairs[dimension]
        v1_data = pair_data.get("v1")
        v2_data = pair_data.get("v2")

        lines.extend([
            f"## {dimension.capitalize()}",
            "",
        ])

        if not v1_data or not v2_data:
            lines.append(f"**⚠️ Unvollständig:** Kein vollständiges v1/v2-Paar gefunden.")
            if pair_data.get("warnings"):
                lines.append("")
                lines.append("**Warnungen:**")
                for warning in pair_data["warnings"]:
                    lines.append(f"- {warning}")
            lines.append("")
            continue

        # Run-IDs
        lines.extend([
            f"**v1 Run:** `{v1_data['run_id']}`",
            f"**v2 Run:** `{v2_data['run_id']}`",
            "",
        ])

        # Bedingungen
        lines.append("**Bedingungen:**")
        lines.append(f"- Dataset: {v1_data['dataset']}")
        lines.append(f"- Model: {v1_data['model']}")
        lines.append(f"- n_used: {v1_data['n_used']}")
        if v1_data.get("seed") is not None:
            lines.append(f"- Seed: {v1_data['seed']}")
        lines.append("")

        # Metriken-Tabelle
        if dimension == "factuality":
            lines.extend([
                "### Metriken",
                "",
                "| Metrik | v1 | v2 | Δ (v2 - v1) |",
                "|--------|----|----|--------------|",
            ])

            for metric in ["precision", "recall", "f1", "balanced_accuracy", "specificity", "mcc", "auroc", "accuracy"]:
                v1_val = v1_data["metrics"].get(metric)
                v2_val = v2_data["metrics"].get(metric)
                delta_val = pair_data.get("delta", {}).get(f"delta_{metric}")

                if v1_val is not None and v2_val is not None:
                    v1_str = f"{v1_val:.4f}" if isinstance(v1_val, float) else str(v1_val)
                    v2_str = f"{v2_val:.4f}" if isinstance(v2_val, float) else str(v2_val)
                    delta_str = f"{delta_val:+.4f}" if delta_val is not None else "N/A"
                    lines.append(f"| {metric} | {v1_str} | {v2_str} | {delta_str} |")

            lines.append("")

            # Confusion Matrix
            if "confusion_matrix" in v1_data["metrics"]:
                cm1 = v1_data["metrics"]["confusion_matrix"]
                cm2 = v2_data["metrics"].get("confusion_matrix", {})
                lines.extend([
                    "### Confusion Matrix",
                    "",
                    "| Version | TP | FP | TN | FN |",
                    "|---------|----|----|----|----|",
                    f"| v1 | {cm1.get('tp', 0)} | {cm1.get('fp', 0)} | {cm1.get('tn', 0)} | {cm1.get('fn', 0)} |",
                    f"| v2 | {cm2.get('tp', 0)} | {cm2.get('fp', 0)} | {cm2.get('tn', 0)} | {cm2.get('fn', 0)} |",
                    "",
                ])

        elif dimension in ("coherence", "readability"):
            lines.extend([
                "### Metriken",
                "",
                "| Metrik | v1 | v2 | Δ (v2 - v1) |",
                "|--------|----|----|--------------|",
            ])

            for metric in ["spearman", "spearman_rho", "pearson", "pearson_r", "mae", "rmse", "r2", "r_squared"]:
                v1_val = v1_data["metrics"].get(metric)
                v2_val = v2_data["metrics"].get(metric)
                delta_val = pair_data.get("delta", {}).get(f"delta_{metric}")

                if v1_val is not None and v2_val is not None:
                    v1_str = f"{v1_val:.4f}" if isinstance(v1_val, float) else str(v1_val)
                    v2_str = f"{v2_val:.4f}" if isinstance(v2_val, float) else str(v2_val)
                    delta_str = f"{delta_val:+.4f}" if delta_val is not None else "N/A"
                    lines.append(f"| {metric} | {v1_str} | {v2_str} | {delta_str} |")

            lines.append("")

        # Hinweise
        if pair_data.get("ignored_candidates"):
            lines.append(f"**Ignorierte Kandidaten:** {len(pair_data['ignored_candidates'])} weitere Runs in derselben Gruppe")
            lines.append("")

        if pair_data.get("warnings"):
            lines.append("**⚠️ Warnungen:**")
            for warning in pair_data["warnings"]:
                lines.append(f"- {warning}")
            lines.append("")
        else:
            lines.append("✅ **Alle Bedingungen identisch außer Prompt-Version**")
            lines.append("")

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Sammelt kapitelrelevante Evaluation-Runs")
    ap.add_argument(
        "--runs-root",
        type=str,
        default="results/evaluation",
        help="Root-Verzeichnis für Runs (default: results/evaluation)",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="results/analysis",
        help="Output-Verzeichnis (default: results/analysis)",
    )

    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanne Runs in: {runs_root}")
    all_runs = find_all_runs(runs_root)
    print(f"Gefunden: {len(all_runs)} kapitelrelevante Runs")

    # Gruppiere nach Bedingungen
    groups = group_runs_by_conditions(all_runs)
    print(f"Gefunden: {len(groups)} Gruppen mit identischen Bedingungen")

    # Wähle beste Paare
    pairs = {}
    incomplete_dimensions = []

    for dimension in ["factuality", "coherence", "readability"]:
        dimension_groups = [g for g in groups.values() if g[0].dimension == dimension]

        if not dimension_groups:
            incomplete_dimensions.append(dimension)
            pairs[dimension] = {
                "v1": None,
                "v2": None,
                "warnings": [f"Keine Runs für {dimension} gefunden"],
            }
            continue

        # Wenn mehrere Gruppen existieren, wähle die mit v1 UND v2
        best_group = None
        for group in dimension_groups:
            v1_count = sum(1 for r in group if r.prompt_version == "v1")
            v2_count = sum(1 for r in group if r.prompt_version == "v2")
            if v1_count > 0 and v2_count > 0:
                best_group = group
                break

        # Fallback: erste Gruppe
        if not best_group:
            best_group = dimension_groups[0]

        v1, v2, ignored = select_best_pair(best_group)

        if not v1 or not v2:
            incomplete_dimensions.append(dimension)
            pairs[dimension] = {
                "v1": asdict(v1) if v1 else None,
                "v2": asdict(v2) if v2 else None,
                "warnings": [f"Unvollständiges Paar: v1={'✓' if v1 else '✗'}, v2={'✓' if v2 else '✗'}"],
                "ignored_candidates": ignored,
            }
            continue

        # Validiere Paar
        is_valid, messages = validate_pair(v1, v2)

        if not is_valid:
            print(f"❌ FEHLER: {dimension} v1/v2-Paar ist inkonsistent:")
            for msg in messages:
                print(f"  - {msg}")
            sys.exit(1)

        # Berechne Delta
        delta = calculate_delta(v1.metrics, v2.metrics)

        pairs[dimension] = {
            "v1": {
                "run_id": v1.run_id,
                "run_path": v1.run_path,
                "dataset": v1.dataset,
                "model": v1.model,
                "seed": v1.seed,
                "n_used": v1.n_used,
                "dataset_signature": v1.dataset_signature,
                "metrics": v1.metrics,
            },
            "v2": {
                "run_id": v2.run_id,
                "run_path": v2.run_path,
                "dataset": v2.dataset,
                "model": v2.model,
                "seed": v2.seed,
                "n_used": v2.n_used,
                "dataset_signature": v2.dataset_signature,
                "metrics": v2.metrics,
            },
            "delta": delta,
            "warnings": [m for m in messages if "warn" in m.lower() or "differ" in m.lower()],
            "ignored_candidates": ignored,
        }

    # Output
    json_path = out_dir / "chapter6_runs.json"
    write_json_output(pairs, json_path)
    print(f"JSON: {json_path}")

    md_path = out_dir / "chapter6_runs.md"
    write_markdown_output(pairs, md_path)
    print(f"Markdown: {md_path}")

    # Zusammenfassung
    print("\n✅ Sammlung abgeschlossen!")
    print(f"Vollständige Paare: {len(pairs) - len(incomplete_dimensions)}/{len(pairs)}")
    if incomplete_dimensions:
        print(f"Unvollständig: {', '.join(incomplete_dimensions)}")


if __name__ == "__main__":
    main()
