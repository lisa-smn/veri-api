#!/bin/bash
# Archiviert Root-Markdown-Dateien nach docs/archive/2026-01-17_cleanup/

set -euo pipefail

ARCHIVE_DIR="docs/archive/2026-01-17_cleanup"
mkdir -p "$ARCHIVE_DIR"

FILES=(
    "PRESENTATION_REPORT.md"
    "PRESENTATION_BEWEISSTUECKE.md"
    "M10_EVALUATION.md"
    "M10_IMPLEMENTATION_SUMMARY.md"
    "M10_TUNING_WORKFLOW.md"
    "QUICKSTART_M10.md"
    "PROJECT_STATUS.md"
    "EVALUATION_DATASETS.md"
    "EVIDENCE_GATE_IMPLEMENTATION.md"
    "EVIDENCE_GATE_ANALYSIS.md"
    "EVIDENCE_GATE_BALANCING.md"
    "EVIDENCE_GATE_BALANCED_APPROACH.md"
    "EVIDENCE_RETRIEVAL_IMPROVEMENTS.md"
    "EXPLAINABILITY_VERIFICATION.md"
    "IMPROVEMENTS_V1.md"
    "PROMPT_AND_FUZZY_IMPROVEMENTS.md"
    "CACHE_FIX.md"
    "ARCHITECTURE_DIAGRAM.md"
    "DATASET_AGENT_VERIFICATION.md"
    "DATASET_REFERENCE_CHECK.md"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "Archiviere: $file"
        git mv "$file" "$ARCHIVE_DIR/" || mv "$file" "$ARCHIVE_DIR/"
    else
        echo "⚠️  Nicht gefunden: $file"
    fi
done

echo "✅ Archivierung abgeschlossen: $ARCHIVE_DIR"

