#!/usr/bin/env python3
"""
Analysiert Evaluation-Runs und identifiziert Prompt-Sensitivität.

Findet alle Runs, gruppiert sie nach ähnlichen Bedingungen (nur Prompt unterschiedlich)
und berechnet Delta-Metriken zwischen Prompt-Varianten.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
import sys
from typing import Any

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@dataclass
class RunRecord:
    """Einzelner Run-Record mit allen Metadaten und Metriken."""

    run_id: str
    run_path: str
    timestamp: str | None = None
    git_commit: str | None = None
    agent: str | None = None  # factuality/coherence/readability
    dimension: str | None = None  # Alias für agent
    dataset: str | None = None
    subset: str | None = None
    manifest_path: str | None = None
    dataset_signature: str | None = None
    n_used: int | None = None
    n_total: int | None = None
    n_failed: int | None = None

    # Prompt identifiers
    prompt_version: str | None = None
    prompt_id: str | None = None
    prompt_hash: str | None = None
    prompt_template_path: str | None = None
    judge_prompt_version: str | None = None

    # Model identifiers
    model: str | None = None
    llm_model: str | None = None  # Alias
    temperature: float | None = None
    seed: int | None = None
    rng_seed: int | None = None  # Alias

    # Metrics (Factuality)
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    specificity: float | None = None
    balanced_accuracy: float | None = None
    bal_acc: float | None = None  # Alias
    mcc: float | None = None
    auroc: float | None = None
    accuracy: float | None = None

    # Metrics (Coherence/Readability)
    spearman: float | None = None
    spearman_rho: float | None = None  # Alias
    pearson: float | None = None
    pearson_r: float | None = None  # Alias
    mae: float | None = None
    rmse: float | None = None
    r2: float | None = None
    r_squared: float | None = None  # Alias

    # Flags
    is_baseline: bool = False
    is_judge: bool = False
    is_confounded: bool = False

    # Debug
    artifact_paths: dict[str, str] = field(default_factory=dict)


def extract_metrics_from_summary(summary_path: Path) -> dict[str, Any]:
    """Extrahiert Metriken aus summary.json."""
    try:
        with summary_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    metrics = {}

    # Factuality metrics
    if "metrics" in data:
        m = data["metrics"]
        # Handle nested dicts with "value" key
        for key in ["precision", "recall", "f1", "specificity", "balanced_accuracy", "mcc", "auroc", "accuracy"]:
            if key in m:
                val = m[key]
                if isinstance(val, dict) and "value" in val:
                    metrics[key] = val["value"]
                elif isinstance(val, (int, float)):
                    metrics[key] = val

        # Aliases
        if "balanced_accuracy" in metrics:
            metrics["bal_acc"] = metrics["balanced_accuracy"]

    # Coherence/Readability metrics
    for key in ["spearman", "pearson", "mae", "rmse", "r2", "r_squared"]:
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


def extract_metadata(run_dir: Path) -> dict[str, Any]:
    """Extrahiert Metadaten aus run_metadata.json oder anderen Quellen."""
    metadata = {}

    # Try run_metadata.json first
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            pass

    # Try summary.json as fallback
    if not metadata:
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            try:
                with summary_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Extract what we can
                    metadata = {
                        "n_used": data.get("n_used"),
                        "n_total": data.get("n_total"),
                        "n_failed": data.get("n_failed"),
                        "dataset_signature": data.get("dataset_signature"),
                    }
            except Exception:
                pass

    return metadata


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
    for key in ["dataset", "data_path", "manifest_path", "manifest"]:
        if key in metadata and metadata[key]:
            val = str(metadata[key])
            if "frank" in val.lower():
                return "FRANK"
            if "sumeval" in val.lower():
                return "SummEval"
            if "finesumfact" in val.lower():
                return "FineSumFact"

    # From path
    path_str = str(run_path).lower()
    if "frank" in path_str:
        return "FRANK"
    if "sumeval" in path_str:
        return "SummEval"
    if "finesumfact" in path_str:
        return "FineSumFact"

    return None


def normalize_prompt_identifier(metadata: dict[str, Any], run_path: Path) -> str | None:
    """Normalisiert Prompt-Identifier (Version, Hash, etc.)."""
    # Try explicit prompt_version (top level)
    if "prompt_version" in metadata and metadata["prompt_version"]:
        return str(metadata["prompt_version"])

    # Try config.prompt_version (most common location)
    config = metadata.get("config", {})
    if "prompt_version" in config and config["prompt_version"]:
        return str(config["prompt_version"])

    # Try prompt_id
    if "prompt_id" in metadata and metadata["prompt_id"]:
        return str(metadata["prompt_id"])

    # Try prompt_hash
    if "prompt_hash" in metadata and metadata["prompt_hash"]:
        return str(metadata["prompt_hash"])

    # Try to infer from run_id
    run_id = metadata.get("run_id", str(run_path.name))
    if "_v" in run_id:
        # Extract version like "v1", "v2", etc.
        parts = run_id.split("_v")
        if len(parts) > 1:
            version_part = parts[1].split("_")[0]
            if version_part and version_part[0] == "v":
                return version_part

    return None


def build_run_record(run_dir: Path) -> RunRecord | None:
    """Baut RunRecord aus Run-Verzeichnis."""
    if not run_dir.is_dir():
        return None

    metadata = extract_metadata(run_dir)
    summary_path = run_dir / "summary.json"
    metrics = extract_metrics_from_summary(summary_path) if summary_path.exists() else {}

    # Build record
    run_id = metadata.get("run_id", run_dir.name)
    dimension = infer_dimension_from_path(run_dir) or metadata.get("dimension") or metadata.get("agent")
    dataset = infer_dataset_from_metadata(metadata, run_dir)
    prompt_version = normalize_prompt_identifier(metadata, run_dir)

    # Model (try multiple keys)
    model = metadata.get("model") or metadata.get("llm_model") or metadata.get("llm_model_name")

    # Seed (try multiple keys)
    seed = metadata.get("seed") or metadata.get("rng_seed")
    if seed is None:
        config = metadata.get("config", {})
        seed = config.get("seed") or config.get("rng_seed")

    # Temperature
    temperature = metadata.get("temperature")
    if temperature is None:
        config = metadata.get("config", {})
        temperature = config.get("temperature")

    # Dataset signature
    dataset_signature = metadata.get("dataset_signature")

    # Manifest/subset
    manifest_path = metadata.get("manifest_path") or metadata.get("manifest")
    subset = metadata.get("subset")

    # Build artifact paths
    artifact_paths = {}
    for fname in ["summary.json", "summary.md", "run_metadata.json", "predictions.jsonl"]:
        fpath = run_dir / fname
        if fpath.exists():
            artifact_paths[fname] = str(fpath)

    record = RunRecord(
        run_id=run_id,
        run_path=str(run_dir),
        timestamp=metadata.get("timestamp"),
        git_commit=metadata.get("git_commit"),
        agent=dimension,
        dimension=dimension,
        dataset=dataset,
        subset=subset,
        manifest_path=manifest_path,
        dataset_signature=dataset_signature,
        n_used=metadata.get("n_used") or metrics.get("n_used"),
        n_total=metadata.get("n_total"),
        n_failed=metadata.get("n_failed"),
        prompt_version=prompt_version,
        model=model,
        llm_model=model,
        temperature=temperature,
        seed=seed,
        rng_seed=seed,
        is_baseline="baseline" in str(run_dir).lower() or "rouge" in str(run_dir).lower() or "bertscore" in str(run_dir).lower(),
        is_judge="judge" in str(run_dir).lower(),
        artifact_paths=artifact_paths,
        **metrics,
    )

    return record


def find_all_runs(runs_root: Path) -> list[RunRecord]:
    """Findet alle Runs rekursiv."""
    records = []

    # Common patterns
    patterns = [
        "**/run_metadata.json",
        "**/summary.json",
    ]

    for pattern in patterns:
        for path in runs_root.rglob(pattern):
            run_dir = path.parent
            # Skip if already processed
            if any(r.run_path == str(run_dir) for r in records):
                continue

            record = build_run_record(run_dir)
            if record:
                records.append(record)

    return records


def group_runs_by_conditions(records: list[RunRecord], strict: bool = False) -> list[dict[str, Any]]:
    """Gruppiert Runs nach ähnlichen Bedingungen (nur Prompt unterschiedlich)."""
    # Filter out baselines and judges (unless comparing judge prompts)
    agent_records = [r for r in records if not r.is_baseline]

    # Group by key conditions
    groups = defaultdict(list)

    for record in agent_records:
        # Build group key (relaxed: dataset_signature optional if manifest_path matches)
        dataset_key = record.dataset_signature
        if not dataset_key:
            dataset_key = record.manifest_path or record.subset or record.dataset

        key_parts = [
            record.dimension,
            record.dataset,
            dataset_key,
            record.model,
            record.temperature if record.temperature is not None else "default",
            record.n_used,
        ]

        if strict:
            key_parts.append(record.seed if record.seed is not None else "no_seed")

        group_key = tuple(key_parts)
        groups[group_key].append(record)

    # Filter groups with multiple prompt versions
    prompt_groups = []
    for group_key, group_records in groups.items():
        if len(group_records) < 2:
            continue

        # Check if prompts differ
        prompt_versions = {r.prompt_version for r in group_records if r.prompt_version}
        if len(prompt_versions) < 2:
            continue

        # Check for confounders
        is_confounded = False
        confounders = []

        if not strict:
            seeds = {r.seed for r in group_records if r.seed is not None}
            if len(seeds) > 1:
                is_confounded = True
                confounders.append("seed")

            temps = {r.temperature for r in group_records if r.temperature is not None}
            if len(temps) > 1:
                is_confounded = True
                confounders.append("temperature")

        prompt_groups.append({
            "group_key": group_key,
            "records": group_records,
            "is_confounded": is_confounded,
            "confounders": confounders,
        })

    return prompt_groups


def calculate_deltas(group: dict[str, Any]) -> list[dict[str, Any]]:
    """Berechnet Delta-Metriken zwischen Prompt-Varianten."""
    records = group["records"]

    # Group by prompt version
    by_prompt = defaultdict(list)
    for record in records:
        prompt_key = record.prompt_version or "unknown"
        by_prompt[prompt_key].append(record)

    # Average metrics per prompt
    prompt_metrics = {}
    for prompt_key, prompt_records in by_prompt.items():
        metrics_avg = defaultdict(list)
        for record in prompt_records:
            for metric_name in [
                "f1", "precision", "recall", "balanced_accuracy", "bal_acc", "mcc", "auroc", "accuracy",
                "spearman", "spearman_rho", "pearson", "pearson_r", "mae", "rmse", "r2", "r_squared",
            ]:
                val = getattr(record, metric_name, None)
                if val is not None:
                    metrics_avg[metric_name].append(val)

        # Average
        prompt_metrics[prompt_key] = {
            k: sum(v) / len(v) if v else None
            for k, v in metrics_avg.items()
        }

    # Calculate deltas (use first prompt as baseline)
    prompt_keys = sorted([k for k in prompt_metrics.keys() if k != "unknown"])
    if len(prompt_keys) < 2:
        return []

    baseline_prompt = prompt_keys[0]
    baseline_metrics = prompt_metrics[baseline_prompt]

    deltas = []
    for alt_prompt in prompt_keys[1:]:
        alt_metrics = prompt_metrics[alt_prompt]
        delta_dict = {
            "baseline_prompt": baseline_prompt,
            "alternative_prompt": alt_prompt,
            "dimension": records[0].dimension,
            "dataset": records[0].dataset,
            "n_used": records[0].n_used,
            "model": records[0].model,
            "is_confounded": group["is_confounded"],
            "confounders": ",".join(group["confounders"]),
        }

        # Calculate deltas for all metrics
        for metric_name in baseline_metrics:
            baseline_val = baseline_metrics[metric_name]
            alt_val = alt_metrics.get(metric_name)
            if baseline_val is not None and alt_val is not None:
                delta = alt_val - baseline_val
                delta_dict[f"baseline_{metric_name}"] = baseline_val
                delta_dict[f"alt_{metric_name}"] = alt_val
                delta_dict[f"delta_{metric_name}"] = delta

        deltas.append(delta_dict)

    return deltas


def write_csv(records: list[RunRecord], output_path: Path):
    """Schreibt Runs als CSV."""
    import csv

    if not records:
        return

    # Get all possible fields (excluding artifact_paths)
    all_fields = set()
    artifact_keys = set()
    for record in records:
        d = asdict(record)
        artifact_paths = d.pop("artifact_paths", {})
        all_fields.update(d.keys())
        artifact_keys.update(artifact_paths.keys())

    # Build fieldnames
    fieldnames = sorted(all_fields)
    # Add artifact fields (only once)
    for key in sorted(artifact_keys):
        fieldnames.append(f"artifact_{key}")

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for record in records:
            row = asdict(record)
            # Flatten artifact_paths
            artifact_paths = row.pop("artifact_paths", {})
            for key, val in artifact_paths.items():
                row[f"artifact_{key}"] = val

            # Convert None to empty string for CSV
            row = {k: (v if v is not None else "") for k, v in row.items()}
            writer.writerow(row)


def write_groups_csv(groups: list[dict[str, Any]], output_path: Path):
    """Schreibt Gruppierungen als CSV."""
    import csv

    all_deltas = []
    for group in groups:
        deltas = calculate_deltas(group)
        all_deltas.extend(deltas)

    if not all_deltas:
        return

    # Get all fields
    all_fields = set()
    for delta in all_deltas:
        all_fields.update(delta.keys())

    fieldnames = sorted(all_fields)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for delta in all_deltas:
            # Convert None to empty string
            row = {k: (v if v is not None else "") for k, v in delta.items()}
            writer.writerow(row)


def write_report(records: list[RunRecord], groups: list[dict[str, Any]], output_path: Path):
    """Schreibt Markdown-Report."""
    from datetime import datetime

    lines = [
        "# Prompt-Sensitivitäts-Analyse",
        "",
        f"**Erstellt:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Zusammenfassung",
        "",
        f"- **Anzahl Runs gescannt:** {len(records)}",
        f"- **Anzahl valider Prompt-Vergleichsgruppen:** {len(groups)}",
        "",
    ]

    # Analyze prompt versions
    prompt_versions = {r.prompt_version for r in records if r.prompt_version}
    if len(prompt_versions) <= 1:
        lines.extend([
            "### Hinweis: Keine unterschiedlichen Prompt-Versionen gefunden",
            "",
            f"Alle gefundenen Runs verwenden die gleiche Prompt-Version: {', '.join(prompt_versions) if prompt_versions else 'unknown'}",
            "",
            "**Um Prompt-Sensitivität zu analysieren, benötigt das System Runs mit unterschiedlichen Prompt-Versionen.**",
            "",
        ])

    # Summary of runs by dimension
    by_dimension = defaultdict(list)
    for record in records:
        if record.dimension:
            by_dimension[record.dimension].append(record)

    if by_dimension:
        lines.extend([
            "## Gefundene Runs (nach Dimension)",
            "",
        ])
        for dim, dim_records in sorted(by_dimension.items()):
            lines.append(f"### {dim.capitalize()} ({len(dim_records)} Runs)")
            lines.append("")
            for record in dim_records[:5]:  # Show first 5
                prompt_info = f"Prompt: {record.prompt_version or 'unknown'}"
                dataset_info = f"Dataset: {record.dataset or 'unknown'}"
                n_info = f"n={record.n_used or '?'}"
                lines.append(f"- `{record.run_id}` - {prompt_info}, {dataset_info}, {n_info}")
            if len(dim_records) > 5:
                lines.append(f"- ... und {len(dim_records) - 5} weitere")
            lines.append("")

    # Calculate deltas for all groups
    all_deltas = []
    for group in groups:
        deltas = calculate_deltas(group)
        all_deltas.extend(deltas)

    if all_deltas:
        lines.extend([
            "",
            "## Top-10 größte Abweichungen",
            "",
        ])

        # Sort by absolute delta (try different metrics)
        for metric in ["delta_f1", "delta_spearman", "delta_spearman_rho", "delta_balanced_accuracy", "delta_bal_acc"]:
            metric_deltas = [d for d in all_deltas if metric in d and d[metric] is not None]
            if metric_deltas:
                metric_deltas.sort(key=lambda x: abs(x[metric]), reverse=True)
                lines.extend([
                    f"### {metric}",
                    "",
                    "| Dimension | Dataset | Baseline → Alt | Δ | Baseline | Alt |",
                    "|-----------|---------|----------------|---|----------|-----|",
                ])

                for delta in metric_deltas[:10]:
                    dim = delta.get("dimension", "")
                    dataset = delta.get("dataset", "")
                    baseline = delta.get("baseline_prompt", "")
                    alt = delta.get("alternative_prompt", "")
                    delta_val = delta.get(metric, 0)
                    baseline_val = delta.get(f"baseline_{metric.replace('delta_', '')}", "")
                    alt_val = delta.get(f"alt_{metric.replace('delta_', '')}", "")

                    lines.append(f"| {dim} | {dataset} | {baseline} → {alt} | {delta_val:+.4f} | {baseline_val} | {alt_val} |")

                lines.append("")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "Diese Analyse identifiziert Runs, die sich nur (oder primär) in der Prompt-Version unterscheiden.",
        "Delta-Metriken zeigen, wie stark sich die Performance bei Prompt-Änderungen verändert.",
        "",
        "**Wichtige Hinweise:**",
        "- Runs mit `is_confounded=True` haben zusätzliche Unterschiede (z.B. Seed, Temperature),",
        "  die die Prompt-Effekte verzerren können.",
        "- Nur Runs mit identischen Bedingungen (Dataset, Model, n_used, etc.) werden verglichen.",
        "- Fehlende Prompt-Identifier werden als 'unknown' markiert und nicht verglichen.",
        "",
    ])

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Analysiert Prompt-Sensitivität in Evaluation-Runs")
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
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Nur 'best case' Gruppen (gleicher Seed erforderlich)",
    )

    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanne Runs in: {runs_root}")
    records = find_all_runs(runs_root)
    print(f"Gefunden: {len(records)} Runs")

    # Write runs CSV
    runs_csv = out_dir / "prompt_sensitivity_runs.csv"
    write_csv(records, runs_csv)
    print(f"Runs CSV: {runs_csv}")

    # Group runs
    print("Gruppiere Runs...")
    groups = group_runs_by_conditions(records, strict=args.strict)
    print(f"Gefunden: {len(groups)} Vergleichsgruppen")

    # Write groups CSV
    groups_csv = out_dir / "prompt_sensitivity_groups.csv"
    write_groups_csv(groups, groups_csv)
    print(f"Groups CSV: {groups_csv}")

    # Write report
    report_md = out_dir / "prompt_sensitivity_report.md"
    write_report(records, groups, report_md)
    print(f"Report: {report_md}")

    print("\n✅ Analyse abgeschlossen!")


if __name__ == "__main__":
    main()
