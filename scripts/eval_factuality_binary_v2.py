"""
Erweiterte Evaluation für FRANK-/FineSumFact-ähnliche Datensätze.

Zusätzlich zur minimalen Version:
- per-example Outputs als JSONL (für Analyse von FP/FN)
- optionales Caching (damit Wiederholungen nicht erneut LLM-Kosten erzeugen)
- robustes Fehlerhandling mit Retries (Run bricht nicht bei einem Beispiel ab)
- flexible Prediction-Regel:
  - via issue_spans threshold (default)
  - oder score cutoff
  - oder kombiniert

Input:
- .jsonl/.csv mit: article, summary, has_error (None wird übersprungen)

Outputs:
- results/<dataset>/run_*.json  (Metriken + Metadaten)
- results/<dataset>/predictions_*.jsonl (jede Zeile: gt/pred/score/num_issues + optional Details)
- results/<dataset>/cache_*.jsonl (optional, falls --cache)
"""

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Any, Optional

from dotenv import load_dotenv

from app.llm.openai_client import OpenAIClient
from app.services.agents.factuality.factuality_agent import FactualityAgent

load_dotenv()


@dataclass
class FrankExample:
    article: str
    summary: str
    has_error: bool
    meta: dict[str, Any] | None = None


def load_jsonl(path: Path) -> List[FrankExample]:
    out: List[FrankExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            has_error = data.get("has_error", None)
            if has_error is None:
                continue
            article = data.get("article", "")
            summary = data.get("summary", "")
            if not isinstance(article, str) or not article.strip():
                continue
            if not isinstance(summary, str) or not summary.strip():
                continue
            out.append(
                FrankExample(
                    article=article.strip(),
                    summary=summary.strip(),
                    has_error=bool(has_error),
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
            if not article or not summary:
                continue

            raw_label = row.get("has_error")
            if raw_label is None:
                continue

            if isinstance(raw_label, str):
                s = raw_label.strip().lower()
                if s in ("1", "true", "yes", "y"):
                    has_error = True
                elif s in ("0", "false", "no", "n"):
                    has_error = False
                else:
                    continue
            else:
                has_error = bool(raw_label)

            out.append(FrankExample(article=article, summary=summary, has_error=has_error))
    return out


def load_dataset(path: Path) -> List[FrankExample]:
    suf = path.suffix.lower()
    if suf == ".jsonl":
        return load_jsonl(path)
    if suf == ".csv":
        return load_csv(path)
    raise ValueError(f"Unsupported dataset format: {path.suffix}. Use .jsonl or .csv.")


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


def _now_tag() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return v


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _cache_key(example: FrankExample, model: str, prompt_version: str) -> str:
    # Minimale deterministische Signatur (nicht kryptografisch)
    # Für echte Stabilität könntest du SHA256 über article+summary bilden.
    a = example.article[:200].replace("\n", " ")
    s = example.summary[:200].replace("\n", " ")
    return f"{model}|{prompt_version}|a:{a}|s:{s}"


def load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    cache: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            k = rec.get("key")
            v = rec.get("value")
            if isinstance(k, str) and isinstance(v, dict):
                cache[k] = v
    return cache


def append_cache(path: Path, key: str, value: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


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
) -> Metrics:
    metrics = Metrics()

    llm_client = OpenAIClient(model_name=llm_model)
    agent = FactualityAgent(llm_client)

    cache = load_cache(cache_path) if use_cache else {}

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    if predictions_path.exists():
        predictions_path.unlink()  # neues File pro Run

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
                    res = agent.run(ex.article, ex.summary, meta={"source": "frank_eval_v2"})
                    result_payload = {
                        "score": res.score,
                        "num_issues": len(res.issue_spans),
                        "issue_spans": [s.model_dump() for s in res.issue_spans],
                        "details": res.details,
                    }
                    if use_cache:
                        append_cache(cache_path, key, result_payload)
                    break
                except Exception as e:
                    last_err = str(e)
                    if attempt < retries:
                        time.sleep(sleep_s)
                    else:
                        metrics.n_failed += 1
                        # schreibe trotzdem prediction record als failed
                        rec = {
                            "index": n - 1,
                            "gt_has_error": ex.has_error,
                            "pred_has_error": None,
                            "failed": True,
                            "error": last_err,
                            "meta": ex.meta,
                        }
                        with predictions_path.open("a", encoding="utf-8") as pf:
                            pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        result_payload = None

        if result_payload is None:
            continue

        score = float(result_payload.get("score", 0.0))
        num_issues = int(result_payload.get("num_issues", 0))

        pred_has_error = decide_pred_has_error(
            score=score,
            num_issues=num_issues,
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
            "gt_has_error": gt,
            "pred_has_error": pred_has_error,
            "score": score,
            "num_issues": num_issues,
            "decision": {
                "mode": decision_mode,
                "issue_threshold": issue_threshold,
                "score_cutoff": score_cutoff,
            },
            "meta": ex.meta,
        }
        with predictions_path.open("a", encoding="utf-8") as pf:
            pf.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if n % 10 == 0:
            print(f"[{n}] GT={gt} PRED={pred_has_error} issues={num_issues} score={score:.2f}")

    return metrics


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


def main():
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
    args = parser.parse_args()

    path = Path(args.dataset_path)
    examples = load_dataset(path)
    if not examples:
        raise SystemExit("Keine Beispiele geladen (oder alle wegen has_error=None/leeren Feldern übersprungen).")

    dataset_name = path.stem
    prompt_version = _env("FACTUALITY_PROMPT_VERSION", "v1") or "v1"

    results_root = Path("results")
    ds_lower = dataset_name.lower()
    if "finesumfact" in ds_lower:
        out_dir = results_root / "finesumfact"
    elif "frank" in ds_lower:
        out_dir = results_root / "frank"
    else:
        out_dir = results_root / "other"

    ts = _now_tag()
    predictions_path = out_dir / f"predictions_{ts}_{args.llm_model}_{prompt_version}.jsonl"
    cache_path = out_dir / f"cache_{args.llm_model}_{prompt_version}.jsonl"

    print(f"Lade {len(examples)} Beispiele aus {path} ...")
    print(f"Decision mode: {args.decision_mode} | issue_threshold={args.issue_threshold} | score_cutoff={args.score_cutoff}")
    print(f"Cache: {'ON' if args.cache else 'OFF'} ({cache_path})")

    metrics = evaluate(
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
    )

    print("\nErgebnisse:")
    print(f"TP: {metrics.tp}  FP: {metrics.fp}  TN: {metrics.tn}  FN: {metrics.fn}  FAILED: {metrics.n_failed}")
    print(f"Accuracy:  {metrics.accuracy:.3f}")
    print(f"Precision: {metrics.precision:.3f}")
    print(f"Recall:    {metrics.recall:.3f}")
    print(f"F1:        {metrics.f1:.3f}")
    print(f"Predictions JSONL: {predictions_path}")

    summary = {
        "timestamp": ts,
        "dataset": dataset_name,
        "model": args.llm_model,
        "prompt_version": prompt_version,
        "num_examples": len(examples) if args.max_examples is None else min(args.max_examples, len(examples)),
        "decision": {
            "mode": args.decision_mode,
            "issue_threshold": args.issue_threshold,
            "score_cutoff": args.score_cutoff,
        },
        "runtime": {
            "retries": args.retries,
            "sleep_s": args.sleep_s,
            "cache": args.cache,
        },
        "metrics": {
            "tp": metrics.tp,
            "fp": metrics.fp,
            "tn": metrics.tn,
            "fn": metrics.fn,
            "failed": metrics.n_failed,
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
        },
        "artifacts": {
            "predictions_jsonl": str(predictions_path),
            "cache_jsonl": str(cache_path) if args.cache else None,
        },
    }

    out_path = save_run_summary(out_dir, summary)
    print(f"Run summary gespeichert: {out_path}")


if __name__ == "__main__":
    main()
