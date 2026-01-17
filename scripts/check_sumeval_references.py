"""Prüft, ob SummEval-Daten Referenzen enthalten."""

import json
from pathlib import Path


def find_reference(row):
    """Sucht nach Referenz-Summary in verschiedenen Feldern."""
    for key in ["ref", "reference", "references", "ref_summary", "gold", "highlights"]:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, list) and val and isinstance(val[0], str):
            return val[0].strip()

    meta = row.get("meta", {})
    for key in ["ref", "reference", "references", "ref_summary", "gold", "highlights"]:
        val = meta.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, list) and val and isinstance(val[0], str):
            return val[0].strip()

    return None


def main():
    data_path = Path("data/sumeval/sumeval_clean.jsonl")
    if not data_path.exists():
        print(f"Datei nicht gefunden: {data_path}")
        return

    total = 0
    refs_found = 0

    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                total += 1
                ref = find_reference(data)
                if ref:
                    refs_found += 1
                    if refs_found == 1:
                        print(f"Erste Referenz gefunden in Beispiel {total}:")
                        print(f"  Keys: {list(data.keys())}")
                        print(f"  Referenz (erste 100 Zeichen): {ref[:100]}...")
            except json.JSONDecodeError:
                continue

    print(f"\nErgebnis: {refs_found}/{total} Beispiele haben Referenzen")
    if refs_found == 0:
        print("❌ Keine Referenzen gefunden - ROUGE/BERTScore nicht möglich")
    else:
        print(f"✅ {refs_found} Referenzen gefunden - ROUGE/BERTScore möglich")


if __name__ == "__main__":
    main()
