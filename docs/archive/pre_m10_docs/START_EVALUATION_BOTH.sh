#!/bin/bash
# Startet Evaluation für beide Datensätze (FRANK + FineSumFact)

cd "$(dirname "$0")"

echo "=========================================="
echo "Factuality Evaluation - Beide Datensätze"
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

# ========== FRANK ==========
echo "=========================================="
echo "1. FRANK Evaluation"
echo "=========================================="
echo ""

CONFIG_FRANK="evaluation_configs/factuality_frank_test_v1.json"
DATASET_FRANK="data/frank/frank_clean.jsonl"

if [ ! -f "$CONFIG_FRANK" ] || [ ! -f "$DATASET_FRANK" ]; then
    echo "Error: FRANK config or dataset not found"
    exit 1
fi

echo "Config: $CONFIG_FRANK"
echo "Dataset: $DATASET_FRANK"
echo "Examples: 300"
echo ""

$PYTHON_CMD scripts/eval_factuality_structured.py "$CONFIG_FRANK" --dataset-path "$DATASET_FRANK"

FRANK_EXIT=$?

if [ $FRANK_EXIT -eq 0 ]; then
    echo ""
    echo "✅ FRANK Evaluation completed"
else
    echo ""
    echo "❌ FRANK Evaluation failed"
    exit $FRANK_EXIT
fi

echo ""
echo "=========================================="
echo "2. FineSumFact Evaluation"
echo "=========================================="
echo ""

# ========== FineSumFact ==========
CONFIG_FINESUMFACT="evaluation_configs/factuality_finesumfact_test_v1.json"
DATASET_FINESUMFACT="data/finesumfact/human_label_test_clean.jsonl"

if [ ! -f "$CONFIG_FINESUMFACT" ] || [ ! -f "$DATASET_FINESUMFACT" ]; then
    echo "Error: FineSumFact config or dataset not found"
    exit 1
fi

echo "Config: $CONFIG_FINESUMFACT"
echo "Dataset: $DATASET_FINESUMFACT"
echo "Examples: 200"
echo ""

$PYTHON_CMD scripts/eval_factuality_structured.py "$CONFIG_FINESUMFACT" --dataset-path "$DATASET_FINESUMFACT"

FINESUMFACT_EXIT=$?

if [ $FINESUMFACT_EXIT -eq 0 ]; then
    echo ""
    echo "✅ FineSumFact Evaluation completed"
else
    echo ""
    echo "❌ FineSumFact Evaluation failed"
    exit $FINESUMFACT_EXIT
fi

# ========== Zusammenfassung ==========
echo ""
echo "=========================================="
echo "✅ Beide Evaluationen abgeschlossen!"
echo "=========================================="
echo ""
echo "📊 Results:"
echo ""
echo "FRANK:"
echo "  - Documentation: results/evaluation/runs/docs/factuality_frank_test_v1.md"
echo "  - Metrics: results/evaluation/runs/results/factuality_frank_test_v1.json"
echo ""
echo "FineSumFact:"
echo "  - Documentation: results/evaluation/runs/docs/factuality_finesumfact_test_v1.md"
echo "  - Metrics: results/evaluation/runs/results/factuality_finesumfact_test_v1.json"
echo ""
echo "📈 Vergleich beider Datensätze:"
echo "  python3 scripts/compare_runs.py --run1 factuality_frank_test_v1 --run2 factuality_finesumfact_test_v1"
echo ""
echo "📊 Kombinierter Run (falls erstellt):"
echo "  - Documentation: results/evaluation/runs/docs/factuality_combined_frank_finesumfact_test_v1.md"
echo ""
echo "=========================================="

exit 0

