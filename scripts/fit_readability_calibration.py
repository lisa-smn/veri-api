"""
Fittet Kalibrierung für Readability-Scores auf einem Calibration-Split.

Input:
  --data data/sumeval/sumeval_clean.jsonl
  --max_examples 500 (Calibration-Split)
  --seed 123 (separater Seed für Calibration)
  --model gpt-4o-mini
  --prompt_version v2

Output:
  results/evaluation/readability/calibration/calibration_params.json

Wichtig: Kein Leakage!
- Calibration-Split: n=500, seed=123
- Test-Split: n=200, seed=42 (wie bisher)
"""

import argparse
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

from dotenv import load_dotenv

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.llm.openai_client import OpenAIClient
from app.services.agents.readability.calibration import (
    HAS_SKLEARN,
    fit_isotonic_calibration,
    fit_linear_calibration,
    save_calibration_params,
)
from app.services.agents.readability.readability_agent import ReadabilityAgent

load_dotenv()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Ungültiges JSONL in {path} @ Zeile {line_no}: {e}") from e
    return rows


def normalize_to_0_1(value: float, min_val: float, max_val: float) -> float:
    """Normalisiert Wert von [min_val, max_val] auf [0, 1]."""
    if max_val <= min_val:
        return 0.0
    normalized = (value - min_val) / (max_val - min_val)
    if normalized < 0.0:
        return 0.0
    if normalized > 1.0:
        return 1.0
    return normalized


def run_calibration_fit(
    rows: list[dict[str, Any]],
    llm_model: str,
    prompt_version: str,
    max_examples: int | None,
    seed: int | None,
    gt_min: float,
    gt_max: float,
    retries: int,
    sleep_s: float,
) -> tuple[list[float], list[float]]:
    """Führt Agent-Runs durch und sammelt Predictions + Ground Truth."""
    if seed is not None:
        random.seed(seed)
        rows = rows.copy()
        random.shuffle(rows)

    if max_examples is not None:
        rows = rows[:max_examples]

    llm = OpenAIClient(model_name=llm_model)
    agent = ReadabilityAgent(llm, prompt_version=prompt_version)

    preds: list[float] = []
    gts: list[float] = []

    print(f"Führe {len(rows)} Agent-Runs durch...")
    for i, row in enumerate(rows):
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)}")

        # Ground Truth
        gt_raw = row.get("gt", {}).get("readability")
        if gt_raw is None:
            gt_raw = row.get("gt", {}).get("fluency")
        if gt_raw is None:
            continue

        gt_norm = normalize_to_0_1(gt_raw, gt_min, gt_max)

        # Agent-Run
        article = row.get("article", "")
        summary = row.get("summary", "")
        if not article or not summary:
            continue

        try:
            result = agent.run(article, summary)
            pred_score = result.score

            preds.append(pred_score)
            gts.append(gt_norm)

            time.sleep(sleep_s)
        except Exception as e:
            print(f"  Fehler bei Beispiel {i}: {e}")
            continue

    print(f"  Gesammelt: {len(preds)} Predictions")
    return preds, gts


def main() -> None:
    ap = argparse.ArgumentParser(description="Fittet Kalibrierung für Readability-Scores")
    ap.add_argument("--data", type=str, required=True, help="Pfad zur JSONL-Datei")
    ap.add_argument(
        "--max_examples",
        type=int,
        default=500,
        help="Anzahl Beispiele für Calibration (default: 500)",
    )
    ap.add_argument(
        "--seed", type=int, default=123, help="Seed für Calibration-Split (default: 123)"
    )
    ap.add_argument(
        "--model", type=str, default="gpt-4o-mini", help="LLM-Modell (default: gpt-4o-mini)"
    )
    ap.add_argument("--prompt_version", type=str, default="v2", help="Prompt-Version (default: v2)")
    ap.add_argument(
        "--out",
        type=str,
        default="results/evaluation/readability/calibration/calibration_params.json",
        help="Output-Pfad für Kalibrierungsparameter",
    )
    ap.add_argument("--gt-min", type=float, default=1.0, help="Minimaler GT-Wert (default: 1.0)")
    ap.add_argument("--gt-max", type=float, default=5.0, help="Maximaler GT-Wert (default: 5.0)")
    ap.add_argument(
        "--retries", type=int, default=1, help="Anzahl Retries bei Fehlern (default: 1)"
    )
    ap.add_argument(
        "--sleep-s",
        type=float,
        default=1.0,
        help="Sleep zwischen Requests in Sekunden (default: 1.0)",
    )
    ap.add_argument(
        "--method",
        type=str,
        default="linear",
        choices=["linear", "isotonic"],
        help="Kalibrierungsmethode (default: linear)",
    )

    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        ap.error(f"Datensatz nicht gefunden: {data_path}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Lade Daten aus: {data_path}")
    rows = load_jsonl(data_path)
    print(f"  {len(rows)} Zeilen geladen")

    print(f"Fitte Kalibrierung mit {args.max_examples} Beispielen (seed={args.seed})...")
    preds, gts = run_calibration_fit(
        rows=rows,
        llm_model=args.model,
        prompt_version=args.prompt_version,
        max_examples=args.max_examples,
        seed=args.seed,
        gt_min=args.gt_min,
        gt_max=args.gt_max,
        retries=args.retries,
        sleep_s=args.sleep_s,
    )

    if len(preds) < 10:
        ap.error(f"Zu wenige Predictions ({len(preds)}), mindestens 10 erforderlich")

    metadata = {
        "llm_model": args.model,
        "prompt_version": args.prompt_version,
        "seed": args.seed,
        "n_samples": len(preds),
        "gt_min": args.gt_min,
        "gt_max": args.gt_max,
        "data_path": str(data_path),
    }

    if args.method == "linear":
        print("Fitte lineare Kalibrierung...")
        a, b = fit_linear_calibration(preds, gts)
        print(f"  a = {a:.6f}, b = {b:.6f}")
        print(f"Speichere Kalibrierungsparameter nach: {out_path}")
        save_calibration_params(
            a=a, b=b, n_samples=len(preds), out_path=out_path, metadata=metadata, method="linear"
        )
    elif args.method == "isotonic":
        if not HAS_SKLEARN:
            ap.error("sklearn ist nicht verfügbar. Bitte installieren: pip install scikit-learn")
        print("Fitte isotonische Kalibrierung...")
        model = fit_isotonic_calibration(preds, gts)
        print(f"  Modell gefittet (n={len(preds)} Beispiele)")
        print(f"Speichere Kalibrierungsparameter nach: {out_path}")
        # Speichere die Trainingsdaten für Rekonstruktion
        save_calibration_params(
            n_samples=len(preds),
            out_path=out_path,
            metadata=metadata,
            method="isotonic",
            isotonic_model=model,
            isotonic_X=preds,
            isotonic_y=gts,
        )

    print("\nKalibrierung abgeschlossen!")


if __name__ == "__main__":
    main()
