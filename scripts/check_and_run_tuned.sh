#!/bin/bash
# Check dependencies and run Tuned FRANK evaluation

set -e

cd "$(dirname "$0")/.."

echo "Checking dependencies..."
if ! python3 -c "import yaml" 2>/dev/null; then
    echo "⚠️  PyYAML not found. Installing..."
    pip3 install PyYAML
fi

echo ""
echo "Running Tuned FRANK evaluation..."
python3 scripts/run_m10_factuality.py configs/m10_factuality_runs.yaml --run-id factuality_frank_tuned_v1

echo ""
echo "Comparing Baseline vs Tuned..."
python3 -c "
import json
from pathlib import Path

baseline_path = Path('results/evaluation/runs/results/factuality_frank_baseline_v1.json')
tuned_path = Path('results/evaluation/runs/results/factuality_frank_tuned_v1.json')

if not tuned_path.exists():
    print('❌ Tuned results not found')
    exit(1)

with baseline_path.open() as f:
    baseline = json.load(f)
with tuned_path.open() as f:
    tuned = json.load(f)

b_metrics = baseline['metrics']
t_metrics = tuned['metrics']

print('='*70)
print('Baseline vs Tuned Comparison')
print('='*70)
print(f\"{'Metric':<20} {'Baseline':<15} {'Tuned':<15} {'Delta':<15}\")
print('-'*70)
print(f\"{'Specificity':<20} {b_metrics['specificity']:<15.3f} {t_metrics['specificity']:<15.3f} {t_metrics['specificity'] - b_metrics['specificity']:+.3f}\")
print(f\"{'Balanced Acc':<20} {b_metrics['balanced_accuracy']:<15.3f} {t_metrics['balanced_accuracy']:<15.3f} {t_metrics['balanced_accuracy'] - b_metrics['balanced_accuracy']:+.3f}\")
print(f\"{'Recall':<20} {b_metrics['recall']:<15.3f} {t_metrics['recall']:<15.3f} {t_metrics['recall'] - b_metrics['recall']:+.3f}\")
print(f\"{'Precision':<20} {b_metrics['precision']:<15.3f} {t_metrics['precision']:<15.3f} {t_metrics['precision'] - b_metrics['precision']:+.3f}\")
print(f\"{'F1':<20} {b_metrics['f1']:<15.3f} {t_metrics['f1']:<15.3f} {t_metrics['f1'] - b_metrics['f1']:+.3f}\")
print('='*70)

# Check criteria
specificity_up = t_metrics['specificity'] > b_metrics['specificity']
bal_acc_up = t_metrics['balanced_accuracy'] > b_metrics['balanced_accuracy']
recall_ok = t_metrics['recall'] > 0.8  # Not completely imploded

print()
if specificity_up and bal_acc_up and recall_ok:
    print('✅ All criteria met:')
    print('   - Specificity increased')
    print('   - Balanced Accuracy increased')
    print('   - Recall acceptable (>0.8)')
    print()
    print('✅ Config can be frozen for FineSumFact evaluation')
else:
    print('⚠️  Some criteria not met:')
    if not specificity_up:
        print('   - Specificity did not increase')
    if not bal_acc_up:
        print('   - Balanced Accuracy did not increase')
    if not recall_ok:
        print(f\"   - Recall too low: {t_metrics['recall']:.3f}\")
"






