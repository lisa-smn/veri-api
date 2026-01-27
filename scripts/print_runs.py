import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Print evaluation run manifests")
    parser.add_argument(
        "--path",
        type=str,
        default="results/evaluation/runs/results",
        help="Path to search for run manifest JSON files (default: results/evaluation/runs/results)",
    )
    parser.add_argument(
        "--docs-path",
        type=str,
        default="results/evaluation/runs/docs",
        help="Optional path to search for run documentation (default: results/evaluation/runs/docs)",
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
        print(f"Path not found: {base_path.absolute()}")
        print(f"Tip: Run an evaluation to generate artifacts in {base_path}")
        return 0

    # Search for run manifest JSON files (exclude _examples.jsonl)
    pattern = "*.json"
    all_paths = sorted(base_path.rglob(pattern))
    paths = [p for p in all_paths if not p.name.endswith("_examples.jsonl")]

    if args.debug:
        print(f"Debug: Searched pattern: {pattern}")
        print(f"Debug: Base path: {base_path.absolute()}")
        print(f"Debug: Found {len(paths)} manifest files (excluded {len(all_paths) - len(paths)} _examples.jsonl):")
        for p in paths:
            print(f"  - {p}")

    if not paths:
        print(f"No run manifests found.")
        print(f"Searched pattern: {pattern} (excluding *_examples.jsonl)")
        print(f"Base path: {base_path.absolute()}")
        print(f"Try: --path <different_path> or check if runs exist in {base_path}")
        return 0

    # Optional docs path
    docs_path = Path(args.docs_path) if args.docs_path else None

    # Print header
    print(f"Found {len(paths)} run manifest(s) (showing up to {args.limit}):")
    print()

    # Print runs
    count = 0
    for p in paths[: args.limit]:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            
            # Extract fields
            run_id = d.get("run_id", "N/A")
            config = d.get("config", {})
            model = config.get("llm_model", "N/A")
            
            # Prefer run_tag over prompt_version (legacy)
            tag = (
                config.get("run_tag")
                or config.get("prompt_version")
                or d.get("run_tag")
                or d.get("prompt_version")
                or "unknown"
            )
            
            dataset = config.get("dataset") or d.get("dataset", "N/A")
            n_examples = d.get("n") or d.get("n_examples") or d.get("n_total", "N/A")
            created_at = d.get("created_at") or config.get("created_at") or "N/A"
            
            # Find related files
            try:
                manifest_path = p.relative_to(Path.cwd())
            except ValueError:
                manifest_path = p
            
            docs_file = None
            examples_file = None
            
            if docs_path and docs_path.exists():
                # Look for corresponding .md file
                docs_candidate = docs_path / f"{p.stem}.md"
                if docs_candidate.exists():
                    try:
                        docs_file = docs_candidate.relative_to(Path.cwd())
                    except ValueError:
                        docs_file = docs_candidate
            
            # Look for examples file
            examples_candidate = p.parent / f"{p.stem}_examples.jsonl"
            if examples_candidate.exists():
                try:
                    examples_file = examples_candidate.relative_to(Path.cwd())
                except ValueError:
                    examples_file = examples_candidate

            # Print run info
            print(f"run_id: {run_id}")
            print(f"  model: {model} | run_tag: {tag} | dataset: {dataset}")
            print(f"  n_examples: {n_examples} | created_at: {created_at}")
            if manifest_path:
                print(f"  manifest: {manifest_path}")
            if docs_file:
                print(f"  docs: {docs_file}")
            if examples_file:
                print(f"  examples: {examples_file}")
            print()
            count += 1
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Warning: Skipping {p} - {e}")
            continue

    if count == 0:
        print("No valid run manifests found (all files had parsing errors).")
        return 1

    if len(paths) > args.limit:
        print(f"... and {len(paths) - args.limit} more (use --limit to show more)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
