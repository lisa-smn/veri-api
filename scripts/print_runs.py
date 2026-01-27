import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Print factuality evaluation runs")
    parser.add_argument(
        "--path",
        type=str,
        default="results",
        help="Base path to search for run JSON files (default: results)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of runs to print (default: 20)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print all searched file paths",
    )
    args = parser.parse_args()

    base_path = Path(args.path)
    if not base_path.exists():
        print(f"Error: Path not found: {base_path}")
        print(f"Searched path: {base_path.absolute()}")
        return 1

    # Search for run JSON files
    pattern = "run_*_gpt-4o-mini_*.json"
    paths = sorted(base_path.rglob(pattern))

    if args.debug:
        print(f"Debug: Searched pattern: {pattern}")
        print(f"Debug: Base path: {base_path.absolute()}")
        print(f"Debug: Found {len(paths)} files:")
        for p in paths:
            print(f"  - {p}")

    if not paths:
        print(f"No runs found.")
        print(f"Searched pattern: {pattern}")
        print(f"Base path: {base_path.absolute()}")
        print(f"Try: --path <different_path> or check if runs exist in {base_path}")
        return 0

    # Print header
    print(f"Found {len(paths)} run(s) (showing up to {args.limit}):")
    print()

    # Print runs
    count = 0
    for p in paths[: args.limit]:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            m = d.get("metrics", {})
            args_dict = d.get("args", {})

            # Prefer run_tag over prompt_version (legacy)
            tag = (
                d.get("run_tag")
                or d.get("prompt_version")
                or d.get("prompt_version_legacy")
                or "unknown"
            )

            print(
                p.name,
                "| dataset:",
                d.get("dataset", "N/A"),
                "| run_tag:",
                tag,
                "| mode:",
                args_dict.get("decision_mode", "N/A"),
                "| thr:",
                args_dict.get("issue_threshold", "N/A"),
                "| cutoff:",
                args_dict.get("score_cutoff", "N/A"),
                "| P/R/F1:",
                round(m.get("precision", 0), 3),
                round(m.get("recall", 0), 3),
                round(m.get("f1", 0), 3),
                "| bal_acc:",
                round(m.get("balanced_accuracy", 0), 3),
            )
            count += 1
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Warning: Skipping {p} - {e}")
            continue

    if count == 0:
        print("No valid runs found (all files had parsing errors).")
        return 1

    if len(paths) > args.limit:
        print(f"\n... and {len(paths) - args.limit} more (use --limit to show more)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
