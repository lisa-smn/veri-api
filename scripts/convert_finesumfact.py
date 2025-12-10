import json
from pathlib import Path

INPUT_PATH = Path("data/finesumfact/machine_label_train.json")
OUTPUT_PATH = Path("data/finesumfact_clean.jsonl")


def load_finesumfact(path: Path) -> list[dict]:
    """Lädt FineSumFact, egal ob als JSON oder JSONL."""
    with path.open("r", encoding="utf-8") as f:
        try:
            obj = json.load(f)
        except json.JSONDecodeError:
            # Fallback: JSONL – jede Zeile ein JSON-Objekt
            f.seek(0)
            records: list[dict] = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
            return records

    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        # zur Not alles in irgendeinem Feld
        for key in ("data", "examples", "items"):
            if key in obj and isinstance(obj[key], list):
                return obj[key]
        return [obj]

    raise TypeError(f"Unerwartete JSON-Struktur: {type(obj)}")


def convert():
    print(f"Lese FineSumFact Rohdaten: {INPUT_PATH}")
    data = load_finesumfact(INPUT_PATH)

    n_total = 0
    n_kept = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as out_f:
        for item in data:
            n_total += 1

            # 1) Artikel
            document = item.get("document") or item.get("doc")
            if not document:
                continue

            # 2) Summary: FineSumFact: model_summary = Liste von Sätzen
            model_summary = item.get("model_summary")

            if isinstance(model_summary, list):
                summary = " ".join(s.strip() for s in model_summary if s and s.strip())
            else:
                summary = model_summary  # falls es doch ein String ist

            if not summary:
                # ohne Summary bringt es nichts
                continue

            # 3) Label: pred_general_factuality_labels = Liste von 0/1 pro Satz
            labels = item.get("pred_general_factuality_labels")
            has_error = None
            if isinstance(labels, list) and labels:
                # True, wenn mindestens ein Satz fehlerhaft ist (1 = Fehler)
                has_error = any(int(x) == 1 for x in labels)

            example = {
                "article": document,
                "summary": summary,
                "has_error": has_error,
            }

            out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
            n_kept += 1

    print(f"Konvertierung fertig. Gelesen: {n_total}, geschrieben: {n_kept}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    convert()
