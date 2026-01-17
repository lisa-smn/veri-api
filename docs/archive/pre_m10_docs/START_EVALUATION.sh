#!/bin/bash
# Startet die erste Factuality-Evaluation

cd "$(dirname "$0")"

echo "=========================================="
echo "Starting Factuality Evaluation"
echo "=========================================="
echo ""

# Prüfe ob Python verfügbar ist
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found. Please install Python 3.8+"
    exit 1
fi

echo "Using Python: $PYTHON_CMD"
echo ""

# Prüfe ob Config existiert
CONFIG="evaluation_configs/factuality_frank_test_v1.json"
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

# Prüfe ob Dataset existiert
DATASET="data/frank/frank_clean.jsonl"
if [ ! -f "$DATASET" ]; then
    echo "Error: Dataset not found: $DATASET"
    exit 1
fi

echo "Config: $CONFIG"
echo "Dataset: $DATASET"
echo ""

# Starte Evaluation
echo "Starting evaluation..."
echo ""

$PYTHON_CMD scripts/eval_factuality_structured.py "$CONFIG" --dataset-path "$DATASET"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Evaluation completed successfully!"
    echo ""
    echo "Results:"
    echo "  - Documentation: results/evaluation/runs/docs/factuality_frank_test_v1.md"
    echo "  - Metrics: results/evaluation/runs/results/factuality_frank_test_v1.json"
    echo "  - Examples: results/evaluation/runs/results/factuality_frank_test_v1_examples.jsonl"
else
    echo "Evaluation failed with exit code $EXIT_CODE"
    echo "Check the logs above for details."
fi
echo "=========================================="

exit $EXIT_CODE

