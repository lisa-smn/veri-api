import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_INPUT_PATH = Path("data1/finesumfact/machine_label_train.json")
DEFAULT_OUTPUT_DIR = Path("data1/finesumfact")


def load_finesumfact(path: Path) -> list[dict[str, Any]]:
    """Lädt FineSumFact, egal ob als JSON (Liste/Dict) oder JSONL (eine Zeile pro Objekt)."""
    with path.open("r", encoding="utf-8") as f:
        try:
            obj = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            out = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
            return out

    # Manche Files sind Dicts mit einem Feld wie "data1"
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        # best-effort: häufige Container-Namen
        for k in ("data1", "examples", "items"):
            if k in obj and isinstance(obj[k], list):
                return obj[k]
        # sonst: single object
        return [obj]
    return []


def _to_int01(x: Any) -> int | None:
    """Versucht, x zu 0/1 zu normalisieren. None wenn unmöglich."""
    if x is None:
        return None
    if isinstance(x, bool):
        return 1 if x else 0
    if isinstance(x, (int, float)):
        try:
            v = int(x)
            return 1 if v == 1 else 0 if v == 0 else None
        except Exception:
            return None
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("1", "true", "yes", "y", "t"):
            return 1
        if s in ("0", "false", "no", "n", "f"):
            return 0
    return None


def _infer_label_source(item: dict[str, Any], input_path: Path) -> str:
    """Human vs Machine heuristisch bestimmen."""
    if "pred_general_factuality_labels" in item:
        return "machine"
    if "label" in item:
        return "human"
    name = input_path.name.lower()
    if "human" in name:
        return "human"
    if "machine" in name:
        return "machine"
    return "unknown"


def _as_text(x: Any) -> str:
    """
    Normalisiert FineSumFact-Felder auf String.
    - str -> strip
    - list[str] -> join
    - list[list[str]] -> flatten + join
    """
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, list):
        parts: list[str] = []
        for item in x:
            if item is None:
                continue
            if isinstance(item, str):
                s = item.strip()
                if s:
                    parts.append(s)
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, str):
                        s2 = sub.strip()
                        if s2:
                            parts.append(s2)
        return " ".join(parts).strip()
    if isinstance(x, dict):
        # falls irgendein Feld verschachtelt kommt
        for k in ("text", "doc", "article", "document", "sentences", "sents"):
            if k in x:
                return _as_text(x[k])
        return ""
    return str(x).strip()


def _extract_article_summary(item: dict[str, Any]) -> tuple[str, str]:
    """Robust: verschiedene Key-Namen + String/Liste abfangen."""
    article_raw = item.get("doc") or item.get("article") or item.get("document") or item.get("text")
    summary_raw = item.get("model_summary") or item.get("summary") or item.get("generated_summary")

    article = _as_text(article_raw)
    summary = _as_text(summary_raw)

    return article, summary


def _extract_sentence_labels(item: dict[str, Any]) -> list[int] | None:
    """
    Liefert Liste 0/1 pro Satz, wenn vorhanden.
    - machine: pred_general_factuality_labels
    - human: label
    """
    labels = None
    if isinstance(item.get("pred_general_factuality_labels"), list):
        labels = item["pred_general_factuality_labels"]
    elif isinstance(item.get("label"), list):
        labels = item["label"]

    if not isinstance(labels, list) or not labels:
        return None

    norm = [_to_int01(x) for x in labels]
    norm = [x for x in norm if x is not None]
    return norm if norm else None


def convert(input_path: Path, output_path: Path, store_sentence_labels: bool = False) -> None:
    print(f"Lese FineSumFact Rohdaten: {input_path}")
    data = load_finesumfact(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_kept = 0
    n_skipped_no_doc = 0
    n_skipped_no_summary = 0
    n_skipped_no_labels = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        for item in data:
            n_total += 1
            if not isinstance(item, dict):
                continue

            article, summary = _extract_article_summary(item)
            if not article:
                n_skipped_no_doc += 1
                continue
            if not summary:
                n_skipped_no_summary += 1
                continue

            sent_labels = _extract_sentence_labels(item)
            if sent_labels is None:
                n_skipped_no_labels += 1
                continue

            # 1 = Fehler
            has_error = any(x == 1 for x in sent_labels)

            label_source = _infer_label_source(item, input_path)
            meta: dict[str, Any] = {
                "dataset": "finesumfact",
                "label_source": label_source,  # human | machine | unknown
                "source": item.get("source"),
                "model": item.get("model"),
                "split": item.get("split"),
                "n_sent_labels": len(sent_labels),
                "n_error_sent": sum(1 for x in sent_labels if x == 1),
            }

            # optional: komplette Satzlabels mitschreiben (kann Output stark vergrößern)
            if store_sentence_labels:
                meta["sentence_labels"] = sent_labels

            example = {
                "article": article,
                "summary": summary,
                "has_error": has_error,
                "meta": meta,
            }

            out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
            n_kept += 1

    print(
        "Konvertierung fertig. "
        f"Gelesen: {n_total}, geschrieben: {n_kept}, "
        f"skip_no_doc: {n_skipped_no_doc}, skip_no_summary: {n_skipped_no_summary}, "
        f"skip_no_labels: {n_skipped_no_labels}"
    )
    print(f"Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert FineSumFact (human or machine labels) to unified JSONL."
    )
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--store-sentence-labels",
        action="store_true",
        help="Store full sentence_labels list in meta (bigger output, useful for analyses).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    if args.output:
        output_path = Path(args.output)
    else:
        # Default: output neben den Rohdaten, mit sprechendem Namen
        stem = input_path.stem
        output_path = DEFAULT_OUTPUT_DIR / f"{stem}_clean.jsonl"

    convert(
        input_path=input_path,
        output_path=output_path,
        store_sentence_labels=args.store_sentence_labels,
    )


if __name__ == "__main__":
    main()
