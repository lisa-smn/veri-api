"""
Erstellt ein gemeinsames FRANK-Subset-Manifest für fairen Vergleich.

Input:
- data/frank/benchmark_data.json (Original mit Referenzen)
- data/frank/human_annotations.json (Gold-Labels)

Output:
- data/frank/frank_subset_manifest.jsonl

Jede Zeile enthält:
- hash, model_name (Identifikation)
- article_text, summary_text, reference (Texte)
- has_error (Gold-Label: bool)
- meta (optional: factuality score, etc.)

Filter:
- Nur Beispiele mit: article + summary + reference + valid annotation
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_manifest_hash(examples: list[dict[str, Any]]) -> str:
    """
    Berechnet SHA256-Hash über alle example_ids für Dataset-Signature.
    """
    ids = sorted([ex.get("hash", "") + "_" + ex.get("model_name", "") for ex in examples])
    content = "\n".join(ids)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def build_manifest(
    benchmark_path: Path,
    annotations_path: Path,
    out_path: Path,
    max_examples: int | None = None,
    seed: int | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """
    Erstellt Manifest aus FRANK benchmark + annotations.

    Returns:
        (examples_list, manifest_hash)
    """
    benchmark = load_json(benchmark_path)
    if not isinstance(benchmark, list):
        raise ValueError(f"benchmark_data.json muss eine Liste sein, erhalten: {type(benchmark)}")

    annotations = load_json(annotations_path)
    if not isinstance(annotations, list):
        raise ValueError(
            f"human_annotations.json muss eine Liste sein, erhalten: {type(annotations)}"
        )

    # Index: (hash, model_name) -> annotation
    ann_index: dict[tuple[str, str], dict[str, Any]] = {}
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        h = ann.get("hash")
        model = ann.get("model_name")
        if h and model:
            ann_index[(str(h), str(model))] = ann

    examples = []
    for item in benchmark:
        if not isinstance(item, dict):
            continue

        article = item.get("article", "").strip()
        summary = item.get("summary", "").strip()
        reference = item.get("reference", "").strip()
        h = item.get("hash")
        model = item.get("model_name")

        # Filter: Alle Felder müssen vorhanden sein
        if not article or not summary or not reference:
            continue

        if not h or not model:
            continue

        # Match annotation
        key = (str(h), str(model))
        ann = ann_index.get(key)

        if ann is None:
            continue  # Skip wenn keine Annotation

        factuality = ann.get("Factuality")
        try:
            factuality_f = float(factuality) if factuality is not None else 1.0
        except (ValueError, TypeError):
            factuality_f = 1.0

        # has_error: factuality < 1.0
        has_error = factuality_f < 1.0

        # id: f"{hash}:{model_name}"
        example_id = f"{h!s}:{model!s}"

        # gold_score: 1.0 wenn not has_error, else 0.0
        gold_score = 1.0 if not has_error else 0.0

        examples.append(
            {
                "id": example_id,
                "hash": str(h),
                "model_name": str(model),
                "article_text": article,
                "summary_text": summary,
                "reference_text": reference,  # reference_text statt reference
                "gold_has_error": has_error,
                "gold_score": gold_score,
                "meta": {
                    "factuality": factuality_f,
                },
            }
        )

    # Optional: Shuffle und limitieren
    if seed is not None:
        import random

        random.seed(seed)
        random.shuffle(examples)

    if max_examples is not None:
        examples = examples[:max_examples]

    # Compute manifest hash (über ids)
    ids = sorted([ex["id"] for ex in examples])
    content = "\n".join(ids)
    manifest_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    # Write manifest
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Write meta.json
    meta_path = out_path.with_suffix(".meta.json")
    meta = {
        "dataset_signature": manifest_hash,
        "n_total": len(examples),
        "n_used": len(examples),  # Alle Beispiele werden verwendet
        "timestamp": datetime.now().isoformat(),
        "source": {
            "benchmark_path": str(benchmark_path),
            "annotations_path": str(annotations_path),
        },
        "filter": {
            "max_examples": max_examples,
            "seed": seed,
        },
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return examples, manifest_hash


def main() -> None:
    ap = argparse.ArgumentParser(description="Erstellt FRANK-Subset-Manifest für fairen Vergleich")
    ap.add_argument(
        "--benchmark",
        type=str,
        default="data/frank/benchmark_data.json",
        help="Pfad zu benchmark_data.json",
    )
    ap.add_argument(
        "--annotations",
        type=str,
        default="data/frank/human_annotations.json",
        help="Pfad zu human_annotations.json",
    )
    ap.add_argument(
        "--output", type=str, default="data/frank/frank_subset_manifest.jsonl", help="Output-Pfad"
    )
    ap.add_argument("--max_examples", type=int, help="Maximale Anzahl Beispiele")
    ap.add_argument("--seed", type=int, help="Random seed für Shuffling")

    args = ap.parse_args()

    benchmark_path = Path(args.benchmark)
    annotations_path = Path(args.annotations)
    out_path = Path(args.output)

    if not benchmark_path.exists():
        ap.error(f"Datei nicht gefunden: {benchmark_path}")
    if not annotations_path.exists():
        ap.error(f"Datei nicht gefunden: {annotations_path}")

    print(f"Erstelle Manifest aus {benchmark_path} und {annotations_path}...")
    examples, manifest_hash = build_manifest(
        benchmark_path=benchmark_path,
        annotations_path=annotations_path,
        out_path=out_path,
        max_examples=args.max_examples,
        seed=args.seed,
    )

    meta_path = out_path.with_suffix(".meta.json")
    print(f"\n✅ Manifest erstellt: {out_path}")
    print(f"   Meta-Datei: {meta_path}")
    print(f"   Beispiele: {len(examples)}")
    print(f"   Dataset-Signature: {manifest_hash}")
    print("\nVerwendung:")
    print(
        f"  - Agent-Eval: python3 scripts/eval_frank_factuality_agent_on_manifest.py --manifest {out_path}"
    )
    print(
        f"  - Baseline-Eval: python3 scripts/eval_frank_factuality_baselines.py --manifest {out_path} --baseline rouge_l"
    )


if __name__ == "__main__":
    main()
