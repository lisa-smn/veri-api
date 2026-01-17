#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Metrics:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def precision(self) -> float | None:
        d = self.tp + self.fp
        return (self.tp / d) if d else None

    @property
    def recall(self) -> float | None:
        d = self.tp + self.fn
        return (self.tp / d) if d else None

    @property
    def f1(self) -> float | None:
        p = self.precision
        r = self.recall
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)

    @property
    def accuracy(self) -> float | None:
        d = self.tp + self.fp + self.tn + self.fn
        return ((self.tp + self.tn) / d) if d else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _latest_predictions(dirpath: Path, pattern: str = "predictions_*.jsonl") -> Path:
    preds = sorted(dirpath.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not preds:
        raise FileNotFoundError(f"No files matching {pattern} in {dirpath}")
    return preds[0]


def _get_score(row: dict[str, Any]) -> float:
    # robuste Defaults: fehlender score zählt als 1.0 (kein Fehler)
    return float(row.get("score", 1.0))


def _get_num_issues(row: dict[str, Any]) -> int:
    # robuste Defaults: fehlender num_issues zählt als 0
    return int(row.get("num_issues", 0))


def _get_gold(row: dict[str, Any]) -> bool | None:
    # FineSumFact/FRANK: meist gt_has_error
    if "gt_has_error" in row:
        return bool(row["gt_has_error"])
    # alternative Namensfälle:
    if "has_error" in row:
        return bool(row["has_error"])
    if "gold" in row and isinstance(row["gold"], dict) and "has_error" in row["gold"]:
        return bool(row["gold"]["has_error"])
    return None


def _get_failed(row: dict[str, Any]) -> bool:
    return bool(row.get("failed", False))


def _example_id(row: dict[str, Any]) -> str:
    # index ist oft vorhanden; fallback auf key/sha
    if "index" in row:
        return f"index={row['index']}"
    if "key" in row:
        return str(row["key"])
    if "sha256" in str(row.get("cache_key", "")):
        return str(row["cache_key"])
    return "<unknown-id>"


def _confusion(rows: list[dict[str, Any]], pred_fn) -> Metrics:
    tp = fp = tn = fn = 0
    for r in rows:
        gold = _get_gold(r)
        if gold is None:
            continue
        pred = bool(pred_fn(r))
        if pred and gold:
            tp += 1
        elif pred and not gold:
            fp += 1
        elif (not pred) and (not gold):
            tn += 1
        else:
            fn += 1
    return Metrics(tp=tp, fp=fp, tn=tn, fn=fn)


def _print_metrics(name: str, m: Metrics) -> None:
    def fmt(x: float | None) -> str:
        return "None" if x is None else f"{x:.3f}"

    print(f"\n== {name} ==")
    print(f"TP={m.tp} FP={m.fp} TN={m.tn} FN={m.fn}")
    print(
        f"precision={fmt(m.precision)} recall={fmt(m.recall)} f1={fmt(m.f1)} acc={fmt(m.accuracy)}"
    )


def _sample_ids(
    rows: list[dict[str, Any]], pred_fn, want_pred: bool, want_gold: bool, k: int
) -> list[str]:
    out: list[str] = []
    for r in rows:
        gold = _get_gold(r)
        if gold is None:
            continue
        pred = bool(pred_fn(r))
        if pred == want_pred and gold == want_gold:
            out.append(_example_id(r))
            if len(out) >= k:
                break
    return out


def report(
    pred_path: Path,
    issue_threshold: int = 1,
    score_cutoff: float = 1.0,
    sample_k: int = 10,
) -> None:
    rows_all = _read_jsonl(pred_path)

    # failed raus
    failed = [r for r in rows_all if _get_failed(r)]
    rows = [r for r in rows_all if not _get_failed(r)]

    golds = [g for g in (_get_gold(r) for r in rows) if g is not None]

    print(f"file: {pred_path}")
    print(f"total lines: {len(rows_all)} | usable: {len(rows)} | failed: {len(failed)}")
    print(f"gold available: {len(golds)} / {len(rows)}")
    if golds:
        print(f"gold error rate: {sum(golds) / len(golds):.3f}")

    # distributions
    score_buckets = Counter(round(_get_score(r), 2) for r in rows)
    issues_buckets = Counter(_get_num_issues(r) for r in rows)
    print("\nTop score buckets:", score_buckets.most_common(10))
    print("Top num_issues buckets:", issues_buckets.most_common(10))

    # anomaly: score<1 & num_issues==0
    anom = [r for r in rows if _get_score(r) < 1.0 and _get_num_issues(r) == 0]
    anom_gold = [r for r in anom if _get_gold(r) is True]
    print(
        f"\nscore<1 & num_issues==0: {len(anom)} / {len(rows)} = {(len(anom) / len(rows) if rows else 0):.3f}"
    )
    if anom:
        share = (len(anom_gold) / len(anom)) if anom else None
        print(f"… davon gold error: {len(anom_gold)} / {len(anom)} = {share:.3f}")

    # decision rules
    pred_issues = lambda r: _get_num_issues(r) >= issue_threshold
    pred_score = lambda r: _get_score(r) < score_cutoff
    pred_either = lambda r: pred_issues(r) or pred_score(r)

    m_issues = _confusion(rows, pred_issues)
    m_score = _confusion(rows, pred_score)
    m_either = _confusion(rows, pred_either)

    _print_metrics(f"issues-only (num_issues>={issue_threshold})", m_issues)
    _print_metrics(f"score-only (score<{score_cutoff})", m_score)
    _print_metrics("either (issues OR score)", m_either)

    # examples
    print("\nExample IDs (first matches):")
    print("TP (pred=1 gold=1):", _sample_ids(rows, pred_either, True, True, sample_k))
    print("FP (pred=1 gold=0):", _sample_ids(rows, pred_either, True, False, sample_k))
    print("FN (pred=0 gold=1):", _sample_ids(rows, pred_either, False, True, sample_k))
    print("TN (pred=0 gold=0):", _sample_ids(rows, pred_either, False, False, sample_k))

    # anomaly examples
    if anom:
        print("\nAnomaly examples (score<1 & issues=0):")
        for r in anom[:sample_k]:
            print(_example_id(r), "gold=", _get_gold(r), "score=", _get_score(r))


def main() -> None:
    ap = argparse.ArgumentParser(description="Quick report for predictions_*.jsonl")
    ap.add_argument(
        "--pred",
        type=str,
        default="",
        help="Path to predictions jsonl. If empty, uses newest in --dir.",
    )
    ap.add_argument(
        "--dir",
        type=str,
        default="results/finesumfact",
        help="Directory to search for latest predictions.",
    )
    ap.add_argument(
        "--issue-threshold",
        type=int,
        default=1,
        help="num_issues threshold for issues-only decision.",
    )
    ap.add_argument(
        "--score-cutoff",
        type=float,
        default=1.0,
        help="score cutoff for score-only decision (pred error if score < cutoff).",
    )
    ap.add_argument("--k", type=int, default=10, help="How many example IDs to print per bucket.")
    args = ap.parse_args()

    if args.pred:
        pred_path = Path(args.pred)
    else:
        pred_path = _latest_predictions(Path(args.dir))

    report(
        pred_path=pred_path,
        issue_threshold=args.issue_threshold,
        score_cutoff=args.score_cutoff,
        sample_k=args.k,
    )


if __name__ == "__main__":
    main()
