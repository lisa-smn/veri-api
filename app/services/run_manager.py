"""
Run-Management-System für strukturierte Evaluation.

Jeder Run wird automatisch dokumentiert:
- Run-Definition (Config)
- Run-Ausführung (Log)
- Ergebnisse (Example-Level + Summary)
- Auswertung (Quant + Robustheit + Subsets)
- Interpretation (Template)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RunDefinition:
    """Frozen Run-Definition - alle Parameter für Reproduzierbarkeit."""

    run_id: str
    dimension: str  # factuality, coherence, readability, all
    dataset: str
    split: str
    llm_model: str
    llm_temperature: float
    llm_seed: int | None
    prompt_versions: dict[str, str]
    explainability_version: str
    thresholds: dict[str, Any]
    max_examples: int | None
    cache_enabled: bool
    description: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def freeze(self) -> dict[str, Any]:
        """Erstellt frozen Config-Dict."""
        return asdict(self)

    def hash(self) -> str:
        """Deterministischer Hash der Run-Definition."""
        data = json.dumps(self.freeze(), sort_keys=True)
        return hashlib.sha256(data.encode("utf-8")).hexdigest()[:12]


@dataclass
class RunExecution:
    """Run-Ausführungs-Log."""

    run_id: str
    definition_hash: str
    started_at: str
    finished_at: str | None = None
    status: str = "running"  # running, success, failed
    num_examples: int = 0
    num_processed: int = 0
    num_failed: int = 0
    num_cached: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def finish(self, status: str = "success"):
        """Markiert Run als beendet."""
        self.finished_at = datetime.now().isoformat()
        self.status = status


@dataclass
class RunResults:
    """Run-Ergebnisse (Example-Level)."""

    run_id: str
    examples: list[dict[str, Any]]
    metrics: dict[str, Any]
    baselines: dict[str, Any] | None = None  # ROUGE, BERTScore, etc.
    explainability_stats: dict[str, Any] | None = None


@dataclass
class RunAnalysis:
    """Quantitative Auswertung."""

    run_id: str
    primary_metrics: dict[str, float]  # F1, Accuracy, etc.
    robustness: dict[str, Any]  # Threshold-Sweeps, etc.
    subsets: dict[str, dict[str, float]]  # Per-Subset Metriken
    error_analysis: dict[str, Any]  # FP/FN Analysis
    baseline_comparison: dict[str, Any] | None = None


@dataclass
class RunDocumentation:
    """Vollständige Run-Dokumentation."""

    run_id: str
    definition: RunDefinition
    execution: RunExecution
    results: RunResults
    analysis: RunAnalysis
    interpretation: str | None = None  # Markdown-Text
    case_studies: list[dict[str, Any]] = field(default_factory=list)


class RunManager:
    """Verwaltet Runs mit automatischer Dokumentation."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.runs_dir = self.base_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        (self.runs_dir / "definitions").mkdir(exist_ok=True)
        (self.runs_dir / "executions").mkdir(exist_ok=True)
        (self.runs_dir / "results").mkdir(exist_ok=True)
        (self.runs_dir / "analyses").mkdir(exist_ok=True)
        (self.runs_dir / "docs").mkdir(exist_ok=True)
        (self.runs_dir / "logs").mkdir(exist_ok=True)

    def create_run(self, definition: RunDefinition) -> RunExecution:
        """Erstellt neuen Run und speichert Definition."""
        run_dir = self.runs_dir / definition.run_id
        run_dir.mkdir(exist_ok=True)

        # Save definition
        def_path = self.runs_dir / "definitions" / f"{definition.run_id}.json"
        with def_path.open("w", encoding="utf-8") as f:
            json.dump(definition.freeze(), f, indent=2, ensure_ascii=False)

        # Create execution log
        execution = RunExecution(
            run_id=definition.run_id,
            definition_hash=definition.hash(),
            started_at=datetime.now().isoformat(),
        )

        exec_path = self.runs_dir / "executions" / f"{definition.run_id}.json"
        with exec_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(execution), f, indent=2, ensure_ascii=False)

        logger.info(f"Created run: {definition.run_id} (hash: {definition.hash()})")
        return execution

    def update_execution(self, execution: RunExecution):
        """Aktualisiert Execution-Log."""
        exec_path = self.runs_dir / "executions" / f"{execution.run_id}.json"
        with exec_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(execution), f, indent=2, ensure_ascii=False)

    def save_results(self, results: RunResults):
        """Speichert Run-Ergebnisse."""
        results_path = self.runs_dir / "results" / f"{results.run_id}.json"

        # Save summary
        summary = {
            "run_id": results.run_id,
            "metrics": results.metrics,
            "num_examples": len(results.examples),
            "baselines": results.baselines,
            "explainability_stats": results.explainability_stats,
        }
        with results_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Save example-level results
        examples_path = self.runs_dir / "results" / f"{results.run_id}_examples.jsonl"
        with examples_path.open("w", encoding="utf-8") as f:
            for ex in results.examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        logger.info(f"Saved results for run: {results.run_id}")

    def save_analysis(self, analysis: RunAnalysis):
        """Speichert Run-Auswertung."""
        analysis_path = self.runs_dir / "analyses" / f"{analysis.run_id}.json"
        with analysis_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(analysis), f, indent=2, ensure_ascii=False)

        logger.info(f"Saved analysis for run: {analysis.run_id}")

    def save_documentation(self, doc: RunDocumentation):
        """Speichert vollständige Run-Dokumentation."""
        doc_path = self.runs_dir / "docs" / f"{doc.run_id}.md"

        # Generate markdown documentation
        md = self._generate_markdown(doc)

        with doc_path.open("w", encoding="utf-8") as f:
            f.write(md)

        logger.info(f"Saved documentation for run: {doc.run_id}")

    def _generate_markdown(self, doc: RunDocumentation) -> str:
        """Generiert Markdown-Dokumentation."""
        lines = [
            f"# Run Documentation: {doc.run_id}",
            "",
            f"**Status:** {doc.execution.status}",
            f"**Created:** {doc.definition.created_at}",
            f"**Started:** {doc.execution.started_at}",
            f"**Finished:** {doc.execution.finished_at or 'N/A'}",
            "",
            "## Run Definition",
            "",
            "```json",
            json.dumps(doc.definition.freeze(), indent=2),
            "```",
            "",
            "## Execution Log",
            "",
            f"- Examples processed: {doc.execution.num_processed}/{doc.execution.num_examples}",
            f"- Failed: {doc.execution.num_failed}",
            f"- Cached: {doc.execution.num_cached}",
            "",
        ]

        if doc.execution.errors:
            lines.extend(
                [
                    "### Errors",
                    "",
                ]
            )
            for err in doc.execution.errors:
                lines.append(f"- {err}")
            lines.append("")

        if doc.execution.warnings:
            lines.extend(
                [
                    "### Warnings",
                    "",
                ]
            )
            for warn in doc.execution.warnings:
                lines.append(f"- {warn}")
            lines.append("")

        lines.extend(
            [
                "## Results",
                "",
                "### Primary Metrics",
                "",
                "```json",
                json.dumps(doc.results.metrics, indent=2),
                "```",
                "",
            ]
        )

        if doc.results.baselines:
            lines.extend(
                [
                    "### Baselines",
                    "",
                    "```json",
                    json.dumps(doc.results.baselines, indent=2),
                    "```",
                    "",
                ]
            )

        lines.extend(
            [
                "## Analysis",
                "",
                "### Primary Metrics",
                "",
                "```json",
                json.dumps(doc.analysis.primary_metrics, indent=2),
                "```",
                "",
            ]
        )

        if doc.analysis.robustness:
            lines.extend(
                [
                    "### Robustness",
                    "",
                    "```json",
                    json.dumps(doc.analysis.robustness, indent=2),
                    "```",
                    "",
                ]
            )

        if doc.analysis.subsets:
            lines.extend(
                [
                    "### Subset Analysis",
                    "",
                    "```json",
                    json.dumps(doc.analysis.subsets, indent=2),
                    "```",
                    "",
                ]
            )

        if doc.interpretation:
            lines.extend(
                [
                    "## Interpretation",
                    "",
                    doc.interpretation,
                    "",
                ]
            )

        if doc.case_studies:
            lines.extend(
                [
                    "## Case Studies",
                    "",
                ]
            )
            for i, case in enumerate(doc.case_studies, 1):
                lines.extend(
                    [
                        f"### Case Study {i}",
                        "",
                        f"**Type:** {case.get('type', 'N/A')}",
                        f"**Example ID:** {case.get('example_id', 'N/A')}",
                        "",
                        case.get("description", ""),
                        "",
                    ]
                )

        return "\n".join(lines)

    def load_run(self, run_id: str) -> RunDocumentation | None:
        """Lädt vollständige Run-Dokumentation."""
        def_path = self.runs_dir / "definitions" / f"{run_id}.json"
        if not def_path.exists():
            return None

        with def_path.open("r", encoding="utf-8") as f:
            def_data = json.load(f)

        definition = RunDefinition(**def_data)

        # Load execution
        exec_path = self.runs_dir / "executions" / f"{run_id}.json"
        execution = RunExecution(run_id=run_id, definition_hash="", started_at="")
        if exec_path.exists():
            with exec_path.open("r", encoding="utf-8") as f:
                exec_data = json.load(f)
                execution = RunExecution(**exec_data)

        # Load results
        results_path = self.runs_dir / "results" / f"{run_id}.json"
        results = None
        if results_path.exists():
            with results_path.open("r", encoding="utf-8") as f:
                results_data = json.load(f)
                examples_path = self.runs_dir / "results" / f"{run_id}_examples.jsonl"
                examples = []
                if examples_path.exists():
                    with examples_path.open("r", encoding="utf-8") as ef:
                        for line in ef:
                            examples.append(json.loads(line))

                results = RunResults(
                    run_id=run_id,
                    examples=examples,
                    metrics=results_data.get("metrics", {}),
                    baselines=results_data.get("baselines"),
                    explainability_stats=results_data.get("explainability_stats"),
                )

        # Load analysis
        analysis_path = self.runs_dir / "analyses" / f"{run_id}.json"
        analysis = None
        if analysis_path.exists():
            with analysis_path.open("r", encoding="utf-8") as f:
                analysis_data = json.load(f)
                analysis = RunAnalysis(**analysis_data)

        return RunDocumentation(
            run_id=run_id,
            definition=definition,
            execution=execution,
            results=results or RunResults(run_id=run_id, examples=[], metrics={}),
            analysis=analysis
            or RunAnalysis(
                run_id=run_id, primary_metrics={}, robustness={}, subsets={}, error_analysis={}
            ),
        )

    def list_runs(self) -> list[str]:
        """Listet alle Run-IDs."""
        def_dir = self.runs_dir / "definitions"
        if not def_dir.exists():
            return []
        return [f.stem for f in def_dir.glob("*.json")]
