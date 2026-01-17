"""
Prüft einen Readability-Run auf Vollständigkeit.

Zeigt:
- Anzahl eindeutiger example_ids
- Anzahl Zeilen in predictions.jsonl
- Ob Judge-Daten vorhanden sind
- Cache-Status
"""

import argparse
from collections import Counter
import json
from pathlib import Path


def check_run(run_dir: Path) -> None:
    """Prüft einen Run."""
    if not run_dir.exists():
        print(f"FEHLER: Run-Verzeichnis nicht gefunden: {run_dir}")
        return

    preds_path = run_dir / "predictions.jsonl"
    summary_path = run_dir / "summary.json"
    metadata_path = run_dir / "run_metadata.json"

    print(f"Run: {run_dir.name}")
    print("=" * 60)

    # Prüfe summary.json
    if summary_path.exists():
        summary = json.load(open(summary_path))
        n_used = summary.get("n_used", 0)
        n_seen = summary.get("n_seen", 0)
        n_failed = summary.get("n_failed", 0)
        cache_mode = summary.get("cache_mode", "unknown")
        cache_hits = summary.get("cache_hits", 0)
        cache_misses = summary.get("cache_misses", 0)

        print("Summary.json:")
        print(f"  n_used: {n_used}")
        print(f"  n_seen: {n_seen}")
        print(f"  n_failed: {n_failed}")
        print(f"  cache_mode: {cache_mode}")
        print(f"  cache_hits: {cache_hits}")
        print(f"  cache_misses: {cache_misses}")
    else:
        print("⚠️  summary.json nicht gefunden")

    # Prüfe predictions.jsonl
    if preds_path.exists():
        ids = []
        judge_count = 0
        agent_count = 0
        total_lines = 0

        with open(preds_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    total_lines += 1
                    example_id = rec.get("example_id")
                    if example_id:
                        ids.append(example_id)
                    if rec.get("pred_judge") is not None:
                        judge_count += 1
                    if rec.get("pred_agent") is not None:
                        agent_count += 1
                except json.JSONDecodeError as e:
                    print(f"⚠️  JSON-Fehler in Zeile {total_lines + 1}: {e}")

        unique_ids = set(ids)
        id_counts = Counter(ids)

        print("\nPredictions.jsonl:")
        print(f"  Total Zeilen: {total_lines}")
        print(f"  Eindeutige example_ids: {len(unique_ids)}")
        print(f"  Mit pred_agent: {agent_count}")
        print(f"  Mit pred_judge: {judge_count}")

        if len(unique_ids) < total_lines:
            print(f"\n⚠️  WARNUNG: Nur {len(unique_ids)} eindeutige IDs bei {total_lines} Zeilen!")
            print("  Häufigste IDs (Duplikate):")
            for id_val, freq in id_counts.most_common(3):
                if freq > 1:
                    print(f"    {id_val}: {freq}x")

        if len(unique_ids) < 200:
            print(
                f"\n❌ FEHLER: Run ist unvollständig! Erwartet: 200 eindeutige IDs, gefunden: {len(unique_ids)}"
            )
        elif len(unique_ids) == 200:
            print(f"\n✅ Run ist vollständig: {len(unique_ids)} eindeutige IDs")
    else:
        print("⚠️  predictions.jsonl nicht gefunden")

    # Prüfe metadata
    if metadata_path.exists():
        metadata = json.load(open(metadata_path))
        print("\nRun Metadata:")
        print(f"  Run ID: {metadata.get('run_id', 'unknown')}")
        print(f"  Git Commit: {metadata.get('git_commit', 'unknown')[:8]}")
        config = metadata.get("config", {})
        print(f"  Cache Mode: {config.get('cache_mode', 'unknown')}")
        print(f"  Judge Prompt: {config.get('judge_prompt_version', 'N/A')}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prüft Readability-Run auf Vollständigkeit")
    ap.add_argument("run_dir", type=str, help="Run-Verzeichnis")

    args = ap.parse_args()

    check_run(Path(args.run_dir))


if __name__ == "__main__":
    main()
