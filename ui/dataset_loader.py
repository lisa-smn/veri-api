"""Dataset Loader für Streamlit UI.

Lädt Beispiele aus verschiedenen Datasets (FRANK, SummEval, etc.)
und stellt einheitliche Mapping-Funktionen bereit.
"""

import json
from pathlib import Path
import random
from typing import Any

# Dataset-Pfade
DATASET_PATHS = {
    "FRANK": Path("data/frank/frank_clean.jsonl"),
    "SummEval": Path("data/sumeval/sumeval_clean.jsonl"),
    "SummaC": Path("data/summac/summac_clean.jsonl"),  # Falls vorhanden
    "SummaCoz": Path("data/summacoz/summacoz_clean.jsonl"),  # Falls vorhanden
    "FineSumFact": Path("data/finesumfact/human_label_test_clean.jsonl"),
}


def get_article_frank(row: dict[str, Any]) -> str | None:
    """Extrahiert Article-Text aus FRANK-Row."""
    article = row.get("article") or row.get("article_text") or ""
    return article.strip() if article else None


def get_summary_frank(row: dict[str, Any]) -> str | None:
    """Extrahiert Summary-Text aus FRANK-Row."""
    summary = row.get("summary") or row.get("summary_text") or ""
    return summary.strip() if summary else None


def get_example_id_frank(row: dict[str, Any], index: int) -> str:
    """Extrahiert Example-ID aus FRANK-Row."""
    meta = row.get("meta", {})
    if isinstance(meta, dict):
        hash_val = meta.get("hash")
        model = meta.get("model_name")
        if hash_val and model:
            return f"{hash_val}_{model}"
    return f"frank_{index}"


def get_article_sumeval(row: dict[str, Any]) -> str | None:
    """Extrahiert Article-Text aus SummEval-Row."""
    article = row.get("article") or row.get("source") or ""
    return article.strip() if article else None


def get_summary_sumeval(row: dict[str, Any]) -> str | None:
    """Extrahiert Summary-Text aus SummEval-Row."""
    summary = row.get("summary") or row.get("hyp") or row.get("system_summary") or ""
    return summary.strip() if summary else None


def get_example_id_sumeval(row: dict[str, Any], index: int) -> str:
    """Extrahiert Example-ID aus SummEval-Row."""
    meta = row.get("meta", {})
    if isinstance(meta, dict):
        doc_id = meta.get("doc_id")
        system = meta.get("system")
        if doc_id:
            return f"{doc_id}_{system}" if system else doc_id
    return f"sumeval_{index}"


# Dataset-Mapper
DATASET_MAPPERS = {
    "FRANK": {
        "get_article": get_article_frank,
        "get_summary": get_summary_frank,
        "get_example_id": get_example_id_frank,
    },
    "SummEval": {
        "get_article": get_article_sumeval,
        "get_summary": get_summary_sumeval,
        "get_example_id": get_example_id_sumeval,
    },
    # Fallback für unbekannte Datasets
    "default": {
        "get_article": lambda row, idx=0: (
            row.get("article") or row.get("article_text") or ""
        ).strip()
        or None,
        "get_summary": lambda row, idx=0: (
            row.get("summary") or row.get("summary_text") or ""
        ).strip()
        or None,
        "get_example_id": lambda row, idx=0: row.get("id")
        or row.get("example_id")
        or f"example_{idx}",
    },
}


def get_dataset_path(dataset_name: str) -> Path | None:
    """Gibt den Pfad zum Dataset zurück, falls vorhanden."""
    path = DATASET_PATHS.get(dataset_name)
    if path and path.exists():
        return path
    return None


def count_lines_jsonl(path: Path, max_check: int = 1000) -> int:
    """Zählt Zeilen in JSONL-Datei (bis max_check)."""
    count = 0
    try:
        with path.open("r", encoding="utf-8") as f:
            for _ in range(max_check):
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    count += 1
    except Exception:
        pass
    return count


def load_random_example(
    dataset_name: str,
    seed: int | None = None,
) -> tuple[str | None, str | None, str | None, dict[str, Any] | None]:
    """
    Lädt ein zufälliges Beispiel aus dem Dataset.

    Returns:
        (article, summary, example_id, metadata) oder (None, None, None, None) bei Fehler
    """
    if seed is not None:
        random.seed(seed)

    path = get_dataset_path(dataset_name)
    if not path:
        return None, None, None, None

    # Mapper auswählen
    mapper = DATASET_MAPPERS.get(dataset_name, DATASET_MAPPERS["default"])
    get_article = mapper["get_article"]
    get_summary = mapper["get_summary"]
    get_example_id = mapper["get_example_id"]

    # Lade alle Zeilen (für kleine Dateien) oder sample
    examples = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for index, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    article = get_article(row)
                    summary = get_summary(row)
                    if article and summary:
                        example_id = get_example_id(row, index)
                        examples.append((article, summary, example_id, row))
                except json.JSONDecodeError:
                    # Log JSON-Fehler, aber weiter machen
                    continue
    except FileNotFoundError:
        raise
    except Exception as e:
        raise ValueError(f"Fehler beim Lesen von {path}: {e}")

    if not examples:
        raise ValueError(f"Keine gültigen Beispiele in {path} gefunden (leer oder alle ungültig)")

    # Zufälliges Beispiel auswählen
    article, summary, example_id, row = random.choice(examples)

    # Validierung: Stelle sicher, dass Strings zurückgegeben werden
    if not isinstance(article, str):
        article = str(article) if article is not None else ""
    if not isinstance(summary, str):
        summary = str(summary) if summary is not None else ""

    if not article or not summary:
        raise ValueError(
            f"Beispiel hat leere Article oder Summary: article_len={len(article) if article else 0}, summary_len={len(summary) if summary else 0}"
        )

    # Metadata extrahieren
    metadata = {
        "dataset": dataset_name,
        "example_id": example_id,
        "index": examples.index((article, summary, example_id, row)),
        "total_examples": len(examples),
    }

    # Füge zusätzliche Metadaten aus row hinzu (falls vorhanden)
    if isinstance(row, dict):
        if "has_error" in row:
            metadata["has_error"] = row["has_error"]
        if "meta" in row and isinstance(row["meta"], dict):
            metadata.update(row["meta"])

    # Zusätzliche Felder aus Row
    if "has_error" in row:
        metadata["has_error"] = row["has_error"]
    if "gt" in row:
        metadata["ground_truth"] = row["gt"]
    if "meta" in row:
        metadata["meta"] = row["meta"]

    return article, summary, example_id, metadata


def get_dataset_info(dataset_name: str) -> dict[str, Any]:
    """
    Gibt Informationen über das Dataset zurück (Anzahl Beispiele, etc.).

    Returns:
        dict mit info oder {"error": "..."}
    """
    path = get_dataset_path(dataset_name)
    if not path:
        return {"error": f"Dataset '{dataset_name}' nicht gefunden"}

    # Zähle Zeilen (nur erste N Zeilen für Performance)
    line_count = count_lines_jsonl(path, max_check=1000)

    return {
        "dataset": dataset_name,
        "path": str(path),
        "exists": True,
        "estimated_examples": line_count if line_count < 1000 else f">{line_count}",
    }
