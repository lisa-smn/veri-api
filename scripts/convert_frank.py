import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_benchmark_data(path: Path) -> list[dict]:
    data = _load_json(path)
    if not isinstance(data, list):
        raise TypeError(f"Benchmark-Datei muss eine Liste sein, erhalten: {type(data)}")
    return data


def load_human_annotations(path: Path) -> dict[tuple[str, str], dict]:
    data = _load_json(path)
    if not isinstance(data, list):
        raise TypeError(f"Annotations-Datei muss eine Liste sein, erhalten: {type(data)}")

    # Lookup: (hash, model_name) -> annotation
    index: dict[tuple[str, str], dict] = {}
    for ann in data:
        if not isinstance(ann, dict):
            continue
        h = ann.get("hash")
        model = ann.get("model_name")
        if not h or not model:
            continue
        index[(str(h), str(model))] = ann
    return index


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def convert_frank(benchmark_path: Path, annotations_path: Path, out_path: Path) -> None:
    benchmark = load_benchmark_data(benchmark_path)
    annotations_index = load_human_annotations(annotations_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_total = 0
    num_written = 0
    num_missing_keys = 0
    num_missing_annotation = 0
    num_skipped_empty_text = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for item in benchmark:
            num_total += 1
            if not isinstance(item, dict):
                num_missing_keys += 1
                continue

            article = item.get("article")
            summary = item.get("summary")

            # Artikel/Summary müssen brauchbar sein
            if (
                not isinstance(article, str)
                or not article.strip()
                or not isinstance(summary, str)
                or not summary.strip()
            ):
                num_skipped_empty_text += 1
                continue

            h = item.get("hash")
            model = item.get("model_name")
            if not h or not model:
                num_missing_keys += 1
                continue

            key = (str(h), str(model))
            ann = annotations_index.get(key)
            if ann is None:
                num_missing_annotation += 1
                continue

            factuality = _as_float(ann.get("Factuality"), default=0.0)

            # Heuristik (dein aktuelles Design):
            # factuality == 1.0 -> kein Fehler
            # factuality < 1.0  -> hat Fehler
            has_error = factuality < 1.0

            record = {
                "article": article.strip(),
                "summary": summary.strip(),
                "has_error": has_error,
                "meta": {
                    "hash": str(h),
                    "model_name": str(model),
                    "factuality": factuality,
                },
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_written += 1

    print(f"Gesamt Benchmark-Einträge: {num_total}")
    print(f"Geschrieben:              {num_written}")
    print(f"Fehlende Keys:            {num_missing_keys}")
    print(f"Fehlende Annotation:      {num_missing_annotation}")
    print(f"Skip leere Texte:         {num_skipped_empty_text}")
    print(f"Ausgabe geschrieben nach: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Konvertiere FRANK nach unified JSONL-Format.")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="data1/frank_benchmark_data.json",
        help="Pfad zu benchmark_data.json",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="data1/frank_human_annotations.json",
        help="Pfad zu human_annotations.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data1/frank_clean.json",
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
