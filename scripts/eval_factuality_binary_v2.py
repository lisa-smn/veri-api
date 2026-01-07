"""
Erweiterte Evaluation für FRANK-/FineSumFact-ähnliche Datensätze (v2).

Features:
- per-example Outputs als JSONL (für Analyse von FP/FN)
- optionales JSONL-Caching (damit Wiederholungen nicht erneut LLM-Kosten erzeugen)
- robustes Fehlerhandling mit Retries (Run bricht nicht bei einem Beispiel ab)
- flexible Prediction-Regel:
  - via issue_spans threshold (default)
  - oder score cutoff
  - oder kombiniert
- stabiler Cache-Key via SHA256 über (model, prompt_version, article, summary)
  -> keine "phantom" Cache-Treffer durch abgeschnittene Texte

Neu (v2.1, Evaluation-seitig):
- "effektive" Issue-Zählung: Filter nach severity und/oder issue_type
  -> reduziert FP-Explosion, wenn ein Prompt zu viele "OTHER/low"-Issues erzeugt
- Speicherung von num_issues_raw (roh) und num_issues (effektiv)

Input:
- .jsonl/.csv mit: article, summary, has_error (None wird übersprungen)

Outputs:
- results/<dataset>/run_*.json  (Metriken + Metadaten)
- results/<dataset>/predictions_*.jsonl (jede Zeile: gt/pred/score/num_issues + Meta)
- results/<dataset>/cache_*.jsonl (optional, falls --cache)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, Dict, Set

from dotenv import load_dotenv

from app.llm.openai_client import OpenAIClient
from app.services.agents.factuality.factuality_agent import FactualityAgent

load_dotenv()


# ----------------------------- Data loading ----------------------------- #


@dataclass
class FrankExample:
    article: str
    summary: str
    has_error: bool
    meta: dict[str, Any] | None = None


def _parse_has_error(raw_label: Any) -> Optional[bool]:
    """
    Robust gegen bool/int/str Labels.
    Erwartet: True/False oder 1/0 oder "true"/"false"/"yes"/"no".
    """
    if raw_label is None:
        return None
    if isinstance(raw_label, bool):
        return raw_label
    if isinstance(raw_label, (int, float)):
        if raw_label == 1:
            return True
        if raw_label == 0:
            return False
        return None
    if isinstance(raw_label, str):
        s = raw_label.strip().lower()
        if s in ("1", "true", "yes", "y", "t"):
            return True
        if s in ("0", "false", "no", "n", "f"):
            return False
        return None
    return None


def load_jsonl(path: Path) -> List[FrankExample]:
    out: List[FrankExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            article = (data.get("article") or "").strip()
            summary = (data.get("summary") or "").strip()
            raw_label = data.get("has_error")
            has_error = _parse_has_error(raw_label)

            # Skip examples ohne Ground Truth / nicht parsebar
            if has_error is None:
                continue
            if not article or not summary:
                continue

            out.append(
                FrankExample(
                    article=article,
                    summary=summary,
                    has_error=has_error,
                    meta=data.get("meta"),
                )
            )
    return out


def load_csv(path: Path) -> List[FrankExample]:
    out: List[FrankExample] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            article = (row.get("article") or "").strip()
            summary = (row.get("summary") or "").strip()
            raw_label = row.get("has_error")

            if not article or not summary:
                continue

            has_error = _parse_has_error(raw_label)
            if has_error is None:
                continue

            out.append(FrankExample(article=article, summary=summary, has_error=has_error))
    return out


def load_dataset(path: Path) -> List[FrankExample]:
    suf = path.suffix.lower()
    if suf == ".jsonl":
        return load_jsonl(path)
    if suf == ".csv":
        return load_csv(path)
    raise ValueError(f"Unsupported dataset format: {path.suffix}. Use .jsonl or .csv.")


# ------------------------------ Metrics ------------------------------ #


@dataclass
class Metrics:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    n_failed: int = 0
    n_skipped: int = 0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def specificity(self) -> float:
        denom = self.tn + self.fp
        return self.tn / denom if denom > 0 else 0.0

    @property
    def balanced_accuracy(self) -> float:
        return (self.recall + self.specificity) / 2.0


# ------------------------------ Helpers ------------------------------ #


def _now_tag() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return v


def _hash_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_key(example: FrankExample, model: str, prompt_version: str) -> str:
    """
    Stabiler Key über den kompletten Input.
    """
    preimage = (
        model
        + "\n"
        + prompt_version
        + "\n<<<ARTICLE>>>\n"
        + example.article
        + "\n<<<SUMMARY>>>\n"
        + example.summary
    )
    return f"sha256:{_hash_sha256(preimage)}"


def load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    cache: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            k = rec.get("key")
            v = rec.get("value")
            if isinstance(k, str) and isinstance(v, dict):
                cache[k] = v
    return cache


def append_cache(path: Path, key: str, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")
        f.flush()


# --- NEW: issue filtering / effective issue counting --- #

_SEVERITY_RANK: Dict[str, int] = {"low": 0, "medium": 1, "high": 2}


def _parse_csv_set(raw: Optional[str]) -> Optional[Set[str]]:
    if raw is None:
        return None
    s = raw.strip()
    if not s:
        return None
    items = [x.strip() for x in s.split(",") if x.strip()]
    if not items:
        return None
    return {x.upper() for x in items}


def count_effective_issues(
    issue_spans: List[dict[str, Any]],
    *,
    severity_min: str = "low",
    allow_types: Optional[Set[str]] = None,
    ignore_types: Optional[Set[str]] = None,
) -> int:
    """
    Zählt Issues für die Decision-Regel.
    Default: zählt alles (wie vorher).
    Optional: Filter nach severity und/oder issue_type.
    """
    if not issue_spans:
        return 0

    min_rank = _SEVERITY_RANK.get((severity_min or "low").lower(), 0)

    n = 0
    for sp in issue_spans:
        sev = (sp.get("severity") or "low").lower()
        rank = _SEVERITY_RANK.get(sev, 0)
        if rank < min_rank:
            continue

        t = sp.get("issue_type")
        t = (t.upper() if isinstance(t, str) and t else "NONE")

        if allow_types is not None and t not in allow_types:
            continue
        if ignore_types is not None and t in ignore_types:
            continue

        n += 1

    return n


def decide_pred_has_error(
    score: float,
    num_issues: int,
    issue_threshold: int,
    score_cutoff: float | None,
    mode: str,
) -> bool:
    # mode: "issues" | "score" | "either" | "both"
    by_issues = num_issues >= issue_threshold
    by_score = (score_cutoff is not None) and (score < score_cutoff)

    if mode == "issues":
        return by_issues
    if mode == "score":
        return bool(by_score)
    if mode == "either":
        return by_issues or bool(by_score)
    if mode == "both":
        return by_issues and bool(by_score)
    raise ValueError(f"Unknown mode: {mode}")


def _compute_metrics_from_rows(
    rows: List[dict[str, Any]],
    *,
    issue_threshold: int,
    score_cutoff: float | None,
    decision_mode: str,
) -> Metrics:
    m = Metrics()
    for r in rows:
        if r.get("failed"):
            m.n_failed += 1
            continue
        gt = bool(r["gt_has_error"])
        score = float(r.get("score", 0.0))
        num_issues = int(r.get("num_issues", 0))  # NOTE: effektiv
        pred = decide_pred_has_error(
            score=score,
            num_issues=num_issues,
            issue_threshold=issue_threshold,
            score_cutoff=score_cutoff,
            mode=decision_mode,
        )

        if gt and pred:
            m.tp += 1
        elif (not gt) and pred:
            m.fp += 1
        elif (not gt) and (not pred):
            m.tn += 1
        else:
            m.fn += 1
    return m


def sweep_thresholds(
    rows: List[dict[str, Any]],
    decision_mode: str,
) -> dict[str, Any]:
    """
    Optionaler Threshold-Sweep, OHNE neue LLM-Aufrufe.
    Nutzt die bereits geloggten (score, num_issues).
    """
    clean = [r for r in rows if not r.get("failed")]
    if not clean:
        return {"note": "no valid rows to sweep"}

    best: Tuple[float, dict[str, Any]] | None = None

    issue_range = range(1, 6)  # 1..5
    score_cutoffs = [round(i / 100, 2) for i in range(50, 100)]  # 0.50..0.99

    issue_candidates = issue_range if decision_mode in ("issues", "either", "both") else [1]
    score_candidates = score_cutoffs if decision_mode in ("score", "either", "both") else [None]

    for it in issue_candidates:
        for sc in score_candidates:
            m = _compute_metrics_from_rows(
                clean,
                issue_threshold=it,
                score_cutoff=sc,
                decision_mode=decision_mode,
            )
            cand = {
                "decision_mode": decision_mode,
                "issue_threshold": it,
                "score_cutoff": sc,
                "metrics": {
                    "tp": m.tp,
                    "fp": m.fp,
                    "tn": m.tn,
                    "fn": m.fn,
                    "accuracy": m.accuracy,
                    "balanced_accuracy": m.balanced_accuracy,
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1": m.f1,
                },
            }
            if best is None or m.f1 > best[0]:
                best = (m.f1, cand)

    return {"best_by_f1": best[1] if best else None, "note": "sweep ranges: issues 1..5, score 0.50..0.99"}


def sweep_thresholds_multi(
    rows: List[dict[str, Any]],
    modes: List[str],
) -> dict[str, Any]:
    """
    Sweep über mehrere decision_modes. Praktisch, wenn man nicht 4 Runs machen will.
    """
    out: Dict[str, Any] = {"by_mode": {}, "best_overall": None}
    best: Tuple[float, dict[str, Any]] | None = None
    for mode in modes:
        info = sweep_thresholds(rows, mode)
        out["by_mode"][mode] = info
        cand = info.get("best_by_f1")
        if cand and isinstance(cand, dict):
            f1 = float(((cand.get("metrics") or {}).get("f1")) or 0.0)
            if best is None or f1 > best[0]:
                best = (f1, cand)
    out["best_overall"] = best[1] if best else None
    return out


# ------------------------------ Evaluation ------------------------------ #


def evaluate(
    examples: Iterable[FrankExample],
    llm_model: str,
    prompt_version: str,
    max_examples: int | None,
    issue_threshold: int,
    score_cutoff: float | None,
    decision_mode: str,
    retries: int,
    sleep_s: float,
    use_cache: bool,
    cache_path: Path,
    predictions_path: Path,
    *,
    cache_minimal: bool = False,
    write_details: bool = False,
    issue_severity_min: str = "low",
    issue_types_allow: Optional[Set[str]] = None,
    issue_types_ignore: Optional[Set[str]] = None,
) -> tuple[Metrics, List[dict[str, Any]]]:
    """
    Führt LLM-Calls (falls nötig) aus und schreibt predictions JSONL.

    Returns:
      - Metrics (für die gewählte Decision-Rule)
      - rows (raw per-example results, für optionalen Sweep / Debug)
    """
    llm_client = OpenAIClient(model_name=llm_model)
    agent = FactualityAgent(llm_client)

    cache = load_cache(cache_path) if use_cache else {}

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    if predictions_path.exists():
        predictions_path.unlink()  # neues File pro Run

    rows: List[dict[str, Any]] = []
    metrics = Metrics()

    n = 0
    for ex in examples:
        if max_examples is not None and n >= max_examples:
            break
        n += 1

        key = _cache_key(ex, llm_model, prompt_version)
        result_payload: dict[str, Any] | None = cache.get(key) if use_cache else None

        if result_payload is None:
            last_err = None
            for attempt in range(retries + 1):
                try:
                    res = agent.run(ex.article, ex.summary, meta={"source": "eval_factuality_binary_v2"})
                    spans = [s.model_dump() for s in res.issue_spans]
                    result_payload = {
                        "score": res.score,
                        "num_issues": len(spans),  # raw (für Backward-Kompatibilität im Cache)
                        "num_issues_raw": len(spans),
                        "issue_spans": spans,
                    }
                    if not cache_minimal:
                        result_payload["details"] = res.details

                    if use_cache:
                        append_cache(cache_path, key, result_payload)
                    break
                except Exception as e:
                    last_err = f"{type(e).__name__}: {e}"
                    if attempt < retries:
                        time.sleep(sleep_s)
                    else:
                        metrics.n_failed += 1
                        rec = {
                            "index": n - 1,
                            "gt_has_error": ex.has_error,
                            "pred_has_error": None,
                            "failed": True,
                            "error": last_err,
                            "meta": ex.meta,
                        }
                        rows.append(rec)
                        with predictions_path.open("a", encoding="utf-8") as pf:
                            pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        result_payload = None

        if result_payload is None:
            continue

        score = float(result_payload.get("score", 0.0))
        issue_spans = result_payload.get("issue_spans") or []
        if not isinstance(issue_spans, list):
            issue_spans = []

        num_issues_raw = int(result_payload.get("num_issues_raw", result_payload.get("num_issues", 0)))
        num_issues_eff = count_effective_issues(
            issue_spans,
            severity_min=issue_severity_min,
            allow_types=issue_types_allow,
            ignore_types=issue_types_ignore,
        )

        pred_has_error = decide_pred_has_error(
            score=score,
            num_issues=num_issues_eff,  # NOTE: effektiv!
            issue_threshold=issue_threshold,
            score_cutoff=score_cutoff,
            mode=decision_mode,
        )
        gt = ex.has_error

        if gt and pred_has_error:
            metrics.tp += 1
        elif (not gt) and pred_has_error:
            metrics.fp += 1
        elif (not gt) and (not pred_has_error):
            metrics.tn += 1
        else:
            metrics.fn += 1

        rec = {
            "index": n - 1,
            "cache_key": key,
            "gt_has_error": gt,
            "pred_has_error": pred_has_error,
            "score": score,
            "num_issues": num_issues_eff,      # effektiv (Decision)
            "num_issues_raw": num_issues_raw,  # roh (Debug/Report)
            "decision": {
                "mode": decision_mode,
                "issue_threshold": issue_threshold,
                "score_cutoff": score_cutoff,
                "issue_severity_min": issue_severity_min,
                "issue_types_allow": sorted(issue_types_allow) if issue_types_allow else None,
                "issue_types_ignore": sorted(issue_types_ignore) if issue_types_ignore else None,
            },
            "meta": ex.meta,
        }

        if write_details:
            rec["issue_spans"] = issue_spans
            if not cache_minimal:
                rec["details"] = result_payload.get("details")

        rows.append(rec)
        with predictions_path.open("a", encoding="utf-8") as pf:
            pf.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if n % 10 == 0:
            print(f"[{n}] GT={gt} PRED={pred_has_error} issues={num_issues_eff} (raw={num_issues_raw}) score={score:.2f}")

    return metrics, rows


def save_run_summary(out_dir: Path, payload: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = payload.get("timestamp") or _now_tag()
    model = payload.get("model", "model")
    pv = payload.get("prompt_version", "v1")
    fn = f"run_{ts}_{model}_{pv}.json".replace(":", "-").replace(" ", "_")
    path = out_dir / fn
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


# ------------------------------ CLI ------------------------------ #


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FactualityAgent on FRANK-like dataset (v2).")
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max-examples", type=int, default=None)

    parser.add_argument("--issue-threshold", type=int, default=1)
    parser.add_argument("--score-cutoff", type=float, default=None)
    parser.add_argument("--decision-mode", type=str, default="issues", choices=["issues", "score", "either", "both"])

    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--sleep-s", type=float, default=1.0)

    parser.add_argument("--cache", action="store_true", help="Enable JSONL cache (recommended).")
    parser.add_argument(
        "--cache-minimal",
        action="store_true",
        help="Store only minimal payload in cache (score/num_issues/issue_spans). Default stores full details too.",
    )
    parser.add_argument(
        "--write-details",
        action="store_true",
        help="Write issue_spans (+ optional details) into predictions JSONL (bigger file, better debugging).",
    )

    # NEW: effective issue counting
    parser.add_argument(
        "--issue-severity-min",
        type=str,
        default="low",
        choices=["low", "medium", "high"],
        help="Count only issue_spans with severity >= this level (default: low = count all).",
    )
    parser.add_argument(
        "--issue-types",
        type=str,
        default=None,
        help="Comma-separated allowlist of issue_type values to count (e.g. ENTITY,NUMBER,DATE). Default: all.",
    )
    parser.add_argument(
        "--ignore-issue-types",
        type=str,
        default=None,
        help="Comma-separated blocklist of issue_type values to ignore (e.g. OTHER). Default: none.",
    )

    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run an offline threshold sweep (NO extra LLM calls) and store best config by F1 in run summary.",
    )
    parser.add_argument(
        "--sweep-modes",
        type=str,
        default=None,
        help="Comma-separated modes to sweep (issues,score,either,both,all). Default: current --decision-mode.",
    )

    args = parser.parse_args()

    path = Path(args.dataset_path)
    examples = load_dataset(path)
    if not examples:
        raise SystemExit("Keine Beispiele geladen (oder alle wegen has_error=None/leeren Feldern übersprungen).")

    dataset_name = path.stem
    prompt_version = _env("FACTUALITY_PROMPT_VERSION", "v1") or "v1"

    results_root = Path("results")
    full_lower = str(path).lower()

    if "finesumfact" in full_lower:
        out_dir = results_root / "finesumfact"
    elif "frank" in full_lower:
        out_dir = results_root / "frank"
    else:
        out_dir = results_root / "other"

    ts = _now_tag()
    predictions_path = out_dir / f"predictions_{ts}_{args.llm_model}_{prompt_version}.jsonl"
    cache_path = out_dir / f"cache_{args.llm_model}_{prompt_version}.jsonl"

    allow_types = _parse_csv_set(args.issue_types)
    ignore_types = _parse_csv_set(args.ignore_issue_types)

    print(f"Lade {len(examples)} Beispiele aus {path} ...")
    print(
        f"Decision mode: {args.decision_mode} | issue_threshold={args.issue_threshold} | score_cutoff={args.score_cutoff}"
    )
    print(
        f"Issue counting: severity_min={args.issue_severity_min} | allow_types={sorted(allow_types) if allow_types else None} | ignore_types={sorted(ignore_types) if ignore_types else None}"
    )
    print(f"Cache: {'ON' if args.cache else 'OFF'} ({cache_path})")
    if args.cache and args.cache_minimal:
        print("Cache-Minimal: ON (details werden NICHT gecached)")

    metrics, rows = evaluate(
        examples=examples,
        llm_model=args.llm_model,
        prompt_version=prompt_version,
        max_examples=args.max_examples,
        issue_threshold=args.issue_threshold,
        score_cutoff=args.score_cutoff,
        decision_mode=args.decision_mode,
        retries=args.retries,
        sleep_s=args.sleep_s,
        use_cache=args.cache,
        cache_path=cache_path,
        predictions_path=predictions_path,
        cache_minimal=args.cache_minimal,
        write_details=args.write_details,
        issue_severity_min=args.issue_severity_min,
        issue_types_allow=allow_types,
        issue_types_ignore=ignore_types,
    )

    sweep_info = None
    if args.sweep:
        modes_raw = (args.sweep_modes or "").strip()
        if not modes_raw:
            modes = [args.decision_mode]
        else:
            if modes_raw.lower() == "all":
                modes = ["issues", "score", "either", "both"]
            else:
                modes = [m.strip() for m in modes_raw.split(",") if m.strip()]
        sweep_info = sweep_thresholds_multi(rows, modes)

    summary = {
        "timestamp": ts,
        "dataset": dataset_name,
        "model": args.llm_model,
        "prompt_version": prompt_version,
        "args": vars(args),
        "counts": {
            "tp": metrics.tp,
            "fp": metrics.fp,
            "tn": metrics.tn,
            "fn": metrics.fn,
            "n_failed": metrics.n_failed,
        },
        "metrics": {
            "accuracy": metrics.accuracy,
            "balanced_accuracy": metrics.balanced_accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
        },
        "sweep": sweep_info,
        "artifacts": {
            "predictions_jsonl": str(predictions_path),
            "cache_jsonl": str(cache_path) if args.cache else None,
        },
    }

    out_path = save_run_summary(out_dir, summary)
    print(f"Run summary gespeichert: {out_path}")


if __name__ == "__main__":
    main()
