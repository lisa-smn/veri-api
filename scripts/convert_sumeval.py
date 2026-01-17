"""
Konvertiert SumEval in ein unified JSONL-Format für Agent-Evaluation.

Output-Format (pro Zeile):
{
  "article": "...",
  "summary": "...",
  "gt": { "readability": <float>, "coherence": <float>, "fluency": <float>, ... },
  "meta": {...}
}

Hinweis:
- SumEval kommt je nach Quelle in leicht unterschiedlichen JSON-Strukturen.
- Dieser Converter ist bewusst defensiv implementiert.
- Normalisierung ist optional und wird NICHT standardmäßig durchgeführt.

HF SummEval Dump (wie bei dir beobachtet):
- article/source: item["source"]
- summary/system output: item["hyp"]
- expert scores: item["expert_coherence"], item["expert_fluency"], ...
"""

import argparse
import json
from pathlib import Path
from typing import Any

# ----------------- helpers ----------------- #


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _to_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _maybe_normalize(x: float | None, normalize_from: float | None) -> float | None:
    """
    Optional: mappt Werte aus [0, normalize_from] auf [0,1].
    Es wird defensiv min=0 angenommen.
    """
    if x is None:
        return None
    if normalize_from is None or normalize_from <= 0:
        return x
    return _clamp01(x / normalize_from)


def _extract_article(item: dict[str, Any]) -> str | None:
    for k in ("article", "source", "document", "doc", "text"):
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _extract_summary(item: dict[str, Any]) -> str | None:
    # HF-Dump nutzt "hyp"
    for k in ("summary", "hyp", "system_output", "decoded", "generated_summary", "hypothesis"):
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _extract_doc_id(item: dict[str, Any]) -> str | None:
    for k in ("id", "doc_id", "document_id", "instance_id"):
        v = item.get(k)
        if v is None:
            continue
        return str(v)
    return None


def _extract_system_name(item: dict[str, Any]) -> str | None:
    # HF-Dump: "model_id" ist i.d.R. vorhanden
    for k in ("system", "model", "model_name", "system_name", "model_id"):
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        # model_id kann auch int sein
        if k == "model_id" and v is not None:
            return str(v)
    return None


def _iter_examples(obj: Any) -> list[dict[str, Any]]:
    # SumEval kann list oder dict mit "data1"/"examples"/"items" sein
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ("data1", "examples", "items"):
            v = obj.get(key)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        return [obj]
    raise TypeError(f"Unerwartete JSON-Struktur: {type(obj)}")


# ----------------- score extraction ----------------- #


def _extract_dimension_score_generic(item: dict[str, Any], dim: str) -> float | None:
    """
    Extrahiert einen Score für eine Dimension (z.B. coherence/readability/fluency)
    aus typischen SumEval-Strukturen (nicht HF-spezifisch).

    Varianten:
    - item[dim] = float
    - item["scores"][dim] = [..] oder float
    - item["annotations"] = [{... dim: x ...}, ...]
    - item["human_scores"][dim] = ...
    """
    if dim in item:
        v = _to_float(item.get(dim))
        if v is not None:
            return v

    for parent in ("scores", "human_scores", "ratings", "eval", "evaluation"):
        block = item.get(parent)
        if isinstance(block, dict):
            v = block.get(dim)
            if isinstance(v, list):
                vals = [_to_float(x) for x in v]
                m = _mean([x for x in vals if x is not None])
                if m is not None:
                    return m
            vv = _to_float(v)
            if vv is not None:
                return vv

    ann = item.get("annotations")
    if isinstance(ann, list) and ann:
        vals: list[float] = []
        for a in ann:
            if not isinstance(a, dict):
                continue
            vals.append(_to_float(a.get(dim)))
        m = _mean([x for x in vals if x is not None])
        if m is not None:
            return m

    return None


def _extract_dim_score(item: dict[str, Any], dim: str) -> float | None:
    """
    Einheitlicher Score-Extractor:
    1) bevorzugt HF 'expert_*' Felder
    2) fällt zurück auf generische Strukturen
    """
    # Mapping: readability im Projekt = fluency im HF-Dump
    aliases: dict[str, list[str]] = {
        "coherence": ["expert_coherence", "coherence"],
        "fluency": ["expert_fluency", "fluency"],
        "readability": ["expert_fluency", "readability", "fluency"],
    }

    for k in aliases.get(dim, [dim]):
        if k in item:
            v = _to_float(item.get(k))
            if v is not None:
                return v

    # fallback: generische Strukturen
    return _extract_dimension_score_generic(item, dim)


# ----------------- conversion ----------------- #


def convert(
    in_path: Path,
    out_path: Path,
    dims: list[str],
    require_dims: list[str],
    normalize_from: float | None,
) -> None:
    obj = _load_json(in_path)
    items = _iter_examples(obj)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_written = 0
    n_skipped = 0

    # overwrite
    with out_path.open("w", encoding="utf-8") as f:
        for item in items:
            n_total += 1

            article = _extract_article(item)
            summary = _extract_summary(item)
            if not article or not summary:
                n_skipped += 1
                continue

            gt: dict[str, float] = {}
            for dim in dims:
                raw_score = _extract_dim_score(item, dim)
                score = _maybe_normalize(raw_score, normalize_from)
                if score is not None:
                    gt[dim] = score

            missing_required = [d for d in require_dims if d not in gt]
            if missing_required:
                n_skipped += 1
                continue

            rec = {
                "article": article,
                "summary": summary,
                "gt": gt,
                "meta": {
                    "doc_id": _extract_doc_id(item),
                    "system": _extract_system_name(item),
                    # HF-Dump Felder (falls vorhanden), als Bonus-Debug:
                    "model_id": item.get("model_id"),
                    "filepath": item.get("filepath"),
                    # Transparenz: readability kommt ggf. aus expert_fluency
                    "readability_source": "expert_fluency" if "readability" in dims else None,
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"SumEval convert done. total={n_total} written={n_written} skipped={n_skipped}")
    print(f"Dimensions: dims={dims} require={require_dims} normalize_from={normalize_from}")
    print(f"Output: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="Path to SumEval JSON")
    ap.add_argument("--output", type=str, default="data1/sumeval_clean.jsonl")

    ap.add_argument(
        "--dims",
        type=str,
        default="coherence,readability,fluency",
        help="Comma-separated dims to extract (e.g. coherence,readability,fluency)",
    )

    ap.add_argument(
        "--require",
        type=str,
        default="readability",
        help="Comma-separated dims that must be present (e.g. readability)",
    )

    ap.add_argument(
        "--normalize_from",
        type=float,
        default=None,
        help="If set (e.g. 5.0), maps score -> score/normalize_from and clamps to [0,1].",
    )

    args = ap.parse_args()
    dims = [d.strip() for d in args.dims.split(",") if d.strip()]
    require_dims = [d.strip() for d in args.require.split(",") if d.strip()]

    convert(
        Path(args.input),
        Path(args.output),
        dims=dims,
        require_dims=require_dims,
        normalize_from=args.normalize_from,
    )


if __name__ == "__main__":
    main()
