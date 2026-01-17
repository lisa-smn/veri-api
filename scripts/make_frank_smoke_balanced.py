#!/usr/bin/env python3
"""
Erstellt einen stratifizierten, balanced Smoke-Datensatz für FRANK.

Input: data/frank/frank_clean.jsonl
Output: data/frank/frank_smoke_balanced_50_seed42.jsonl

Verteilung: 25 pos / 25 neg (bei ungerader N: neg bekommt den Rest)
Deterministisch: seed=42
"""

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any


def load_examples(input_path: Path, label_key: str = "has_error") -> list[dict[str, Any]]:
    """Lädt Beispiele aus JSONL und validiert Label-Feld."""
    examples = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                if label_key not in ex:
                    print(
                        f"⚠️  WARNING: Label-Feld '{label_key}' nicht gefunden in Beispiel. Verfügbare Keys: {list(ex.keys())}"
                    )
                    continue
                examples.append(ex)
            except json.JSONDecodeError as e:
                print(f"⚠️  WARNING: JSON-Decode-Fehler: {e}", file=sys.stderr)
                continue
    return examples


def stratify_examples(
    examples: list[dict[str, Any]], n: int, seed: int, label_key: str = "has_error"
) -> list[dict[str, Any]]:
    """
    Stratifiziert Beispiele: n/2 pos, n/2 neg (bei ungerader N: neg bekommt den Rest).

    Returns:
        Stratifizierte Liste der Länge n (oder weniger, falls Dataset zu klein).
    """
    # Trenne nach Label
    pos_examples = [ex for ex in examples if ex.get(label_key) is True]
    neg_examples = [ex for ex in examples if ex.get(label_key) is False]

    pos_count = len(pos_examples)
    neg_count = len(neg_examples)

    print(f"Dataset-Verteilung: pos={pos_count}, neg={neg_count}, total={len(examples)}")

    if pos_count == 0 or neg_count == 0:
        print(
            f"❌ FEHLER: Single-class Dataset (pos={pos_count}, neg={neg_count})", file=sys.stderr
        )
        print(
            "   Metriken wie Balanced Accuracy, MCC, AUROC sind nicht definiert.", file=sys.stderr
        )
        sys.exit(1)

    # Berechne Target-Verteilung
    n_pos = n // 2
    n_neg = n - n_pos  # Bei ungerader N: neg bekommt den Rest

    if pos_count < n_pos:
        print(
            f"⚠️  WARNING: Nur {pos_count} positive Beispiele verfügbar, benötigt {n_pos}. Verwende {pos_count}.",
            file=sys.stderr,
        )
        n_pos = pos_count
        n_neg = min(n - n_pos, neg_count)
    if neg_count < n_neg:
        print(
            f"⚠️  WARNING: Nur {neg_count} negative Beispiele verfügbar, benötigt {n_neg}. Verwende {neg_count}.",
            file=sys.stderr,
        )
        n_neg = neg_count
        n_pos = min(n - n_neg, pos_count)

    if n_pos == 0 or n_neg == 0:
        print(
            f"❌ FEHLER: Nach Anpassung: pos={n_pos}, neg={n_neg}. Kann keinen balanced Datensatz erstellen.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Deterministische Zufallsauswahl
    rng = random.Random(seed)
    selected_pos = rng.sample(pos_examples, n_pos)
    selected_neg = rng.sample(neg_examples, n_neg)

    # Kombiniere und mische (deterministisch)
    selected = selected_pos + selected_neg
    rng.shuffle(selected)

    return selected


def main():
    ap = argparse.ArgumentParser(
        description="Erstellt einen stratifizierten, balanced Smoke-Datensatz für FRANK."
    )
    ap.add_argument(
        "--input",
        type=str,
        default="data/frank/frank_clean.jsonl",
        help="Input JSONL-Datei (default: data/frank/frank_clean.jsonl)",
    )
    ap.add_argument(
        "--output",
        type=str,
        default="data/frank/frank_smoke_balanced_50_seed42.jsonl",
        help="Output JSONL-Datei (default: data/frank/frank_smoke_balanced_50_seed42.jsonl)",
    )
    ap.add_argument(
        "--n",
        type=int,
        default=50,
        help="Anzahl Beispiele (default: 50, wird stratifiziert: n/2 pos, n/2 neg)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed für deterministische Auswahl (default: 42)",
    )
    ap.add_argument(
        "--label_key",
        type=str,
        default="has_error",
        help="Label-Feld im JSON (default: has_error)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Überschreibe Output-Datei falls vorhanden (default: False)",
    )

    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"❌ FEHLER: Input-Datei nicht gefunden: {input_path}", file=sys.stderr)
        sys.exit(1)

    if output_path.exists() and not args.force:
        print(f"⚠️  WARNING: Output-Datei existiert bereits: {output_path}", file=sys.stderr)
        print(
            "   Überschreibe nicht automatisch. Bitte manuell löschen, anderen Namen wählen, oder --force verwenden.",
            file=sys.stderr,
        )
        sys.exit(1)

    if output_path.exists() and args.force:
        print(f"⚠️  Überschreibe existierende Datei: {output_path}")

    # Lade Beispiele
    print(f"Lade Beispiele aus: {input_path}")
    examples = load_examples(input_path, label_key=args.label_key)

    if len(examples) == 0:
        print("❌ FEHLER: Keine gültigen Beispiele gefunden.", file=sys.stderr)
        sys.exit(1)

    # Stratifiziere
    print(f"Stratifiziere auf n={args.n} (seed={args.seed})...")
    selected = stratify_examples(examples, args.n, args.seed, label_key=args.label_key)

    # Zähle finale Verteilung
    pos_final = sum(1 for ex in selected if ex.get(args.label_key) is True)
    neg_final = sum(1 for ex in selected if ex.get(args.label_key) is False)

    # Schreibe Output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for ex in selected:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print("\n✅ Smoke-Datensatz erstellt:")
    print(f"   Output: {output_path}")
    print(f"   Beispiele: {len(selected)}")
    print(f"   Verteilung: pos={pos_final}, neg={neg_final}")
    print("\n📊 Verifikation (jq):")
    print(f"   jq -r '.{args.label_key}' {output_path} | sort | uniq -c")


if __name__ == "__main__":
    main()
