#!/bin/bash
# Startet FineSumFact-Evaluation

cd "$(dirname "$0")"

echo "=========================================="
echo "FineSumFact Evaluation - Bachelorarbeit"
echo "=========================================="
echo ""

# Prüfe Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found"
    exit 1
fi

echo "Using Python: $PYTHON_CMD"
echo ""

# Config
CONFIG="evaluation_configs/factuality_finesumfact_test_v1.json"
DATASET="data/finesumfact/human_label_test_clean.jsonl"

# Prüfe Config
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config not found: $CONFIG"
    exit 1
fi

# Prüfe Dataset
if [ ! -f "$DATASET" ]; then
    echo "Error: Dataset not found: $DATASET"
    exit 1
fi

echo "Config: $CONFIG"
echo "Dataset: $DATASET"
echo "Examples: 200 (max_examples in config)"
echo "Cache: Enabled"
echo ""

# Prüfe Cache
CACHE="results/evaluation/factuality/cache_finesumfact_gpt-4o-mini_v3_uncertain_spans.jsonl"
if [ -f "$CACHE" ]; then
    CACHE_SIZE=$(wc -l < "$CACHE" 2>/dev/null || echo "0")
    echo "Cache gefunden: $CACHE_SIZE Einträge"
    echo "→ Bereits evaluierte Beispiele werden übersprungen"
else
    echo "Kein Cache vorhanden - alle Beispiele werden neu evaluiert"
fi
echo ""

# Starte Evaluation
echo "Starting FineSumFact evaluation..."
echo ""

$PYTHON_CMD scripts/eval_factuality_structured.py "$CONFIG" --dataset-path "$DATASET"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ FineSumFact Evaluation completed successfully!"
    echo ""
    echo "📊 Results:"
    echo "  - Documentation: results/evaluation/runs/docs/factuality_finesumfact_test_v1.md"
    echo "  - Metrics: results/evaluation/runs/results/factuality_finesumfact_test_v1.json"
    echo "  - Examples: results/evaluation/runs/results/factuality_finesumfact_test_v1_examples.jsonl"
    echo "  - Analysis: results/evaluation/runs/analyses/factuality_finesumfact_test_v1.json"
    echo ""
    echo "📈 Vergleich mit FRANK:"
    echo "  python3 scripts/compare_runs.py --run1 factuality_frank_test_v1 --run2 factuality_finesumfact_test_v1"
    echo ""
    echo "📈 Quick view:"
    echo "  cat results/evaluation/runs/docs/factuality_finesumfact_test_v1.md | head -50"
else
    echo "❌ Evaluation failed with exit code $EXIT_CODE"
    echo "Check the output above for details."
fi
echo "=========================================="

exit $EXIT_CODE

