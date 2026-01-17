#!/bin/bash
# Helper-Script: Findet die neuesten Readability-Runs mit seed42

BASE_DIR="results/evaluation/readability"

echo "Neueste Readability-Runs (seed42):"
ls -td ${BASE_DIR}/readability_*_seed42 2>/dev/null | head -5 | while read dir; do
    if [ -f "${dir}/predictions.jsonl" ]; then
        echo "  ${dir}"
        # Prüfe auf Judge-Daten
        if head -1 "${dir}/predictions.jsonl" 2>/dev/null | grep -q "pred_judge"; then
            echo "    -> Enthält Judge-Daten"
        else
            echo "    -> Keine Judge-Daten"
        fi
    fi
done

echo ""
echo "Verwendung:"
echo "  python3 scripts/compare_agent_vs_judge.py \\"
echo "    --run_dir_agent <agent_run_dir> \\"
echo "    --run_dir_judge <judge_run_dir> \\"
echo "    --out docs/status_pack/2026-01-08/judge_agreement_readability.md"

