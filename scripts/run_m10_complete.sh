#!/bin/bash
# Kompletter M10-Evaluation-Workflow

cd "$(dirname "$0")/.."

echo "=========================================="
echo "M10 Factuality Evaluation - Complete Workflow"
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

echo "Step 1: Run Baseline"
echo "-------------------"
$PYTHON_CMD scripts/run_m10_factuality.py --run-id factuality_frank_baseline_v1

if [ $? -ne 0 ]; then
    echo "❌ Baseline failed"
    exit 1
fi

echo ""
echo "Step 2: Analyze Baseline & Suggest Tuning"
echo "-----------------------------------------"
$PYTHON_CMD scripts/tune_from_baseline.py --baseline-run-id factuality_frank_baseline_v1

echo ""
echo "💡 Review suggestions above, then update configs/m10_factuality_runs.yaml"
echo "   Press Enter to continue with remaining runs..."
read

echo ""
echo "Step 3: Run Remaining Runs"
echo "-------------------------"
$PYTHON_CMD scripts/run_m10_factuality.py --skip-baseline

if [ $? -ne 0 ]; then
    echo "❌ Some runs failed"
    exit 1
fi

echo ""
echo "Step 4: Aggregate Results"
echo "------------------------"
$PYTHON_CMD scripts/aggregate_m10_results.py

echo ""
echo "=========================================="
echo "✅ M10 Evaluation Complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - Summary Matrix: results/evaluation/summary_matrix.csv"
echo "  - Summary Report: results/evaluation/summary.md"
echo "  - Individual Runs: results/evaluation/runs/docs/*.md"
echo ""






