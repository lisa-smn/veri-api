import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from dotenv import load_dotenv

from app.llm.openai_client import OpenAIClient
from services.agents.factuality.factuality_agent import FactualityAgent

load_dotenv()


@dataclass
class FrankExample:
    article: str
    summary: str
    has_error: bool  # Ground truth: enthält die Summary faktische Fehler?


def load_jsonl(path: Path) -> List[FrankExample]:
    examples: List[FrankExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            article = data["article"]
            summary = data["summary"]
            has_error = bool(data["has_error"])
            examples.append(
                FrankExample(article=article, summary=summary, has_error=has_error)
            )
    return examples


def load_csv(path: Path) -> List[FrankExample]:
    examples: List[FrankExample] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            article = row["article"]
            summary = row["summary"]
            raw_label = row["has_error"]

            if isinstance(raw_label, str):
                raw_label_lower = raw_label.strip().lower()
                if raw_label_lower in ("1", "true", "yes", "y"):
                    has_error = True
                else:
                    has_error = False
            else:
                has_error = bool(raw_label)

            examples.append(
                FrankExample(article=article, summary=summary, has_error=has_error)
            )
    return examples


def load_dataset(path: Path) -> List[FrankExample]:
    if path.suffix.lower() == ".jsonl":
        return load_jsonl(path)
    elif path.suffix.lower() == ".csv":
        return load_csv(path)
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}. Use .jsonl or .csv.")


@dataclass
class Metrics:
    tp: int
    fp: int
    tn: int
    fn: int

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


def evaluate_frank(
    examples: Iterable[FrankExample],
    llm_model: str = "gpt-4o-mini",
    max_examples: int | None = None,
    error_threshold: int = 1,
) -> Metrics:
    """
    Einfacher Evaluationslauf:
    - Ground truth: has_error (True/False)
    - Prediction: agent findet >= error_threshold Fehler (num_errors)
    """

    llm_client = OpenAIClient(model_name=llm_model)
    agent = FactualityAgent(llm_client)

    tp = fp = tn = fn = 0

    for i, ex in enumerate(examples):
        if max_examples is not None and i >= max_examples:
            break

        result = agent.run(ex.article, ex.summary, meta={"source": "frank_eval"})

        num_errors = result.details.get("num_errors", 0)
        pred_has_error = num_errors >= error_threshold

        gt = ex.has_error

        if gt and pred_has_error:
            tp += 1
        elif not gt and pred_has_error:
            fp += 1
        elif not gt and not pred_has_error:
            tn += 1
        elif gt and not pred_has_error:
            fn += 1

        # Kleine Fortschrittsanzeige
        if (i + 1) % 10 == 0:
            print(
                f"[{i+1}] GT={gt} PRED={pred_has_error} "
                f"num_errors={num_errors} score={result.score:.2f}"
            )

    return Metrics(tp=tp, fp=fp, tn=tn, fn=fn)


def save_run_results(
    metrics: dict,
    dataset_name: str,
    out_dir: Path,
    model_name: str,
    prompt_version: str,
    temperature: float | None,
    seed: int | None,
    num_examples: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    payload = {
        "timestamp": timestamp,
        "dataset": dataset_name,
        "model": model_name,
        "prompt_version": prompt_version,
        "temperature": temperature,
        "seed": seed,
        "num_examples": num_examples,
        "metrics": metrics,
    }

    filename = f"run_{timestamp}_{model_name}_{prompt_version}.json"
    filename = filename.replace(":", "-").replace(" ", "_")

    out_path = out_dir / filename

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nRun-Ergebnisse gespeichert in: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FactualityAgent on FRANK-like dataset."
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to FRANK-like dataset (.jsonl or .csv)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o-mini",
        help="Name of the LLM model to use (passed to OpenAIClient).",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional: limit number of evaluated examples.",
    )
    parser.add_argument(
        "--error-threshold",
        type=int,
        default=1,
        help="Agent prediction: 'has_error' wenn num_errors >= threshold.",
    )
    args = parser.parse_args()

    path = Path(args.dataset_path)
    examples = load_dataset(path)

    print(f"Lade {len(examples)} Beispiele aus {path} ...")

    metrics_obj = evaluate_frank(
        examples,
        llm_model=args.llm_model,
        max_examples=args.max_examples,
        error_threshold=args.error_threshold,
    )

    # Konsolenausgabe
    print("\nErgebnisse:")
    print(f"TP: {metrics_obj.tp}  FP: {metrics_obj.fp}  TN: {metrics_obj.tn}  FN: {metrics_obj.fn}")
    print(f"Accuracy:  {metrics_obj.accuracy:.3f}")
    print(f"Precision: {metrics_obj.precision:.3f}")
    print(f"Recall:    {metrics_obj.recall:.3f}")
    print(f"F1:        {metrics_obj.f1:.3f}")

    # Für Speicherung als dict aufbereiten
    metrics_dict = {
        "tp": metrics_obj.tp,
        "fp": metrics_obj.fp,
        "tn": metrics_obj.tn,
        "fn": metrics_obj.fn,
        "accuracy": metrics_obj.accuracy,
        "precision": metrics_obj.precision,
        "recall": metrics_obj.recall,
        "f1": metrics_obj.f1,
    }

    # Metadaten für den Run
    dataset_name = path.stem  # z.B. 'finesumfact_clean'
    model_name = args.llm_model
    prompt_version = os.getenv("FACTUALITY_PROMPT_VERSION", "v1")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    seed = int(os.getenv("LLM_SEED", "0"))

    # Effektive Anzahl Beispiele
    if args.max_examples is None:
        num_examples = len(examples)
    else:
        num_examples = min(args.max_examples, len(examples))

    # Output-Verzeichnis abhängig vom Datensatz
    results_root = Path("results")
    ds_lower = dataset_name.lower()
    if "finesumfact" in ds_lower:
        out_dir = results_root / "finesumfact"
    elif "frank" in ds_lower:
        out_dir = results_root / "frank"
    else:
        out_dir = results_root / "other"

    save_run_results(
        metrics=metrics_dict,
        dataset_name=dataset_name,
        out_dir=out_dir,
        model_name=model_name,
        prompt_version=prompt_version,
        temperature=temperature,
        seed=seed,
        num_examples=num_examples,
    )


if __name__ == "__main__":
    main()
