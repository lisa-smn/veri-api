"""
Tool zum Vergleich von Evaluation-Runs.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.run_manager import RunManager


def compare_runs(run_id1: str, run_id2: str, base_dir: Path):
    """Vergleicht zwei Runs."""
    manager = RunManager(base_dir)

    run1 = manager.load_run(run_id1)
    run2 = manager.load_run(run_id2)

    if not run1:
        print(f"Error: Run {run_id1} not found")
        return
    if not run2:
        print(f"Error: Run {run_id2} not found")
        return

    print("=" * 80)
    print("RUN COMPARISON")
    print("=" * 80)
    print(f"\nRun 1: {run_id1}")
    print(f"Run 2: {run_id2}")
    print()

    # Compare definitions
    print("Definition Comparison:")
    print(f"  Hash 1: {run1.definition.hash()}")
    print(f"  Hash 2: {run2.definition.hash()}")
    if run1.definition.hash() == run2.definition.hash():
        print("  → Identical configurations")
    else:
        print("  → Different configurations")
        # Show differences
        def1 = run1.definition.freeze()
        def2 = run2.definition.freeze()
        for key in set(def1.keys()) | set(def2.keys()):
            val1 = def1.get(key)
            val2 = def2.get(key)
            if val1 != val2:
                print(f"    {key}: {val1} → {val2}")
    print()

    # Compare metrics
    print("Metrics Comparison:")
    metrics1 = run1.results.metrics if run1.results else {}
    metrics2 = run2.results.metrics if run2.results else {}

    for key in set(metrics1.keys()) | set(metrics2.keys()):
        val1 = metrics1.get(key, 0)
        val2 = metrics2.get(key, 0)
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            diff = val2 - val1
            print(f"  {key:20s} {val1:8.3f} → {val2:8.3f} ({diff:+.3f})")
        else:
            print(f"  {key:20s} {val1} → {val2}")
    print()

    # Compare execution
    print("Execution Comparison:")
    print(f"  Status:     {run1.execution.status} → {run2.execution.status}")
    print(f"  Processed:  {run1.execution.num_processed} → {run2.execution.num_processed}")
    print(f"  Failed:     {run1.execution.num_failed} → {run2.execution.num_failed}")
    print(f"  Cached:     {run1.execution.num_cached} → {run2.execution.num_cached}")
    print()

    # Analysis comparison
    if run1.analysis and run2.analysis:
        print("Analysis Comparison:")
        print(
            f"  Best Threshold 1: {run1.analysis.robustness.get('best_threshold', {}).get('threshold', 'N/A')}"
        )
        print(
            f"  Best Threshold 2: {run2.analysis.robustness.get('best_threshold', {}).get('threshold', 'N/A')}"
        )
        print(f"  FP 1: {run1.analysis.error_analysis.get('num_fp', 0)}")
        print(f"  FP 2: {run2.analysis.error_analysis.get('num_fp', 0)}")
        print(f"  FN 1: {run1.analysis.error_analysis.get('num_fn', 0)}")
        print(f"  FN 2: {run2.analysis.error_analysis.get('num_fn', 0)}")
    print()

    print("=" * 80)


def list_runs(base_dir: Path):
    """Listet alle verfügbaren Runs."""
    manager = RunManager(base_dir)
    runs = manager.list_runs()

    if not runs:
        print("No runs found")
        return

    print(f"Found {len(runs)} runs:\n")
    for run_id in sorted(runs):
        run = manager.load_run(run_id)
        if run:
            status = run.execution.status
            metrics = run.results.metrics if run.results else {}
            f1 = metrics.get("f1", 0)
            print(f"  {run_id:40s} {status:10s} F1: {f1:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Compare evaluation runs")
    parser.add_argument("--list", action="store_true", help="List all runs")
    parser.add_argument("--run1", type=str, help="First run ID")
    parser.add_argument("--run2", type=str, help="Second run ID")
    parser.add_argument("--base-dir", type=str, default=None, help="Base directory for runs")

    args = parser.parse_args()

    if args.base_dir:
        base_dir = Path(args.base_dir)
    else:
        base_dir = ROOT / "results" / "evaluation"

    if args.list:
        list_runs(base_dir)
    elif args.run1 and args.run2:
        compare_runs(args.run1, args.run2, base_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
