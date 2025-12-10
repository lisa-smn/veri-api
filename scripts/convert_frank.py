import json
import argparse
from pathlib import Path


def load_benchmark_data(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_human_annotations(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # baue ein Lookup: (hash, model_name) -> annotation
    index = {}
    for ann in data:
        h = ann.get("hash")
        model = ann.get("model_name")
        if h is None or model is None:
            continue
        index[(h, model)] = ann
    return index


def convert_frank(benchmark_path: Path, annotations_path: Path, out_path: Path):
    benchmark = load_benchmark_data(benchmark_path)
    annotations_index = load_human_annotations(annotations_path)

    num_total = 0
    num_matched = 0
    num_missing = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for item in benchmark:
            num_total += 1

            article = item.get("article", "")
            summary = item.get("summary", "")

            h = item.get("hash")
            model = item.get("model_name")

            if h is None or model is None:
                num_missing += 1
                continue

            ann = annotations_index.get((h, model))
            if ann is None:
                num_missing += 1
                continue

            factuality = ann.get("Factuality", 0.0)

            try:
                factuality = float(factuality)
            except Exception:
                factuality = 0.0

            # einfache Heuristik:
            # Factuality = 1.0 -> keine Fehler
            # sonst: Summary enthält Fehler
            has_error = factuality < 1.0

            record = {
                "article": article,
                "summary": summary,
                "has_error": has_error,
                "meta": {
                    "hash": h,
                    "model_name": model,
                    "factuality": factuality,
                },
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_matched += 1

    print(f"Gesamt Benchmark-Einträge: {num_total}")
    print(f"Mit Annotation gematcht:   {num_matched}")
    print(f"Ohne Annotation:          {num_missing}")
    print(f"Ausgabe geschrieben nach: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Konvertiere FRANK nach unified JSONL-Format.")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="data/frank_benchmark_data.json",
        help="Pfad zu benchmark_data.json",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="data/frank_human_annotations.json",
        help="Pfad zu human_annotations.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/frank_clean.jsonl",
        help="Zieldatei im JSONL-Format",
    )
    args = parser.parse_args()

    convert_frank(
        Path(args.benchmark),
        Path(args.annotations),
        Path(args.output),
    )


if __name__ == "__main__":
    main()
