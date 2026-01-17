"""
Berechnet klassische Baseline-Metriken für Readability/Coherence-Evaluation auf SummEval.

Input:
- data/sumeval/sumeval_clean.jsonl
- Falls Referenzen vorhanden: ROUGE-1/2/L, BERTScore
- Falls keine Referenzen: Nur Readability-Formeln (Flesch, Flesch-Kincaid, Gunning Fog)

Output:
- results/evaluation/baselines/<run_id>/
  - predictions.jsonl
  - summary.json
  - summary.md
  - run_metadata.json

Metriken:
- Pearson r, Spearman ρ, MAE, RMSE, R² (mit Bootstrap-CIs)
- Vergleich Baseline-Scores vs gt.readability/gt.coherence
"""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import random
import re
import subprocess
import sys
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Optional imports für Baselines
try:
    from rouge_score import rouge_scorer

    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False

try:
    from bert_score import score as bert_score

    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False

try:
    from sacrebleu import BLEU

    HAS_BLEU = True
except ImportError:
    HAS_BLEU = False

try:
    import nltk
    from nltk.translate.meteor_score import meteor_score

    HAS_METEOR = True
except ImportError:
    HAS_METEOR = False

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------
# IO helpers
# ---------------------------


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


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def get_git_commit() -> str | None:
    """Versucht Git-Commit-Hash zu ermitteln."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def find_reference(row: dict[str, Any]) -> str | None:
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


# ---------------------------
# Metrics
# ---------------------------


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (denx * deny) if denx > 0 and deny > 0 else 0.0


def _rank(values: list[float]) -> list[float]:
    idx = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[idx[j + 1]] == values[idx[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[idx[k]] = avg_rank
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    if not xs:
        return 0.0
    return pearson(_rank(xs), _rank(ys))


def mae(xs: list[float], ys: list[float]) -> float:
    if not xs:
        return 0.0
    return sum(abs(x - y) for x, y in zip(xs, ys)) / len(xs)


def rmse(xs: list[float], ys: list[float]) -> float:
    if not xs:
        return 0.0
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(xs, ys)) / len(xs))


def r_squared(xs: list[float], ys: list[float]) -> float:
    if not xs:
        return 0.0
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - x) ** 2 for x, y in zip(xs, ys))
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def bootstrap_ci(
    metric_func,
    xs: list[float],
    ys: list[float],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> dict[str, float]:
    if not xs or not ys or len(xs) != len(ys):
        return {"median": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    rng = random.Random(seed)
    n = len(xs)
    resamples: list[float] = []

    for _ in range(n_resamples):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        xs_resampled = [xs[i] for i in indices]
        ys_resampled = [ys[i] for i in indices]
        metric_value = metric_func(xs_resampled, ys_resampled)
        resamples.append(metric_value)

    resamples.sort()
    alpha = 1.0 - confidence
    lower_idx = int(n_resamples * (alpha / 2))
    upper_idx = int(n_resamples * (1 - alpha / 2))

    return {
        "median": resamples[len(resamples) // 2],
        "ci_lower": resamples[lower_idx],
        "ci_upper": resamples[upper_idx],
    }


def normalize_to_0_1(x: float, min_v: float, max_v: float) -> float:
    if max_v <= min_v:
        raise ValueError("gt_max muss > gt_min sein")
    if x < min_v:
        x = min_v
    if x > max_v:
        x = max_v
    return (x - min_v) / (max_v - min_v)


# ---------------------------
# Readability Formeln
# ---------------------------


def count_syllables(word: str) -> int:
    """Einfache Silbenzählung für englische Wörter."""
    word = word.lower().strip()
    if not word:
        return 0
    if word.endswith("e"):
        word = word[:-1]
    vowels = "aeiouy"
    syllable_count = 0
    prev_was_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_was_vowel:
            syllable_count += 1
        prev_was_vowel = is_vowel
    return max(1, syllable_count)


def flesch_reading_ease(text: str) -> float:
    """
    Flesch Reading Ease Score (0-100, höher = leichter).
    Normalisiert auf [0,1]: score / 100
    """
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.5  # Neutral

    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.5

    total_syllables = sum(count_syllables(word) for word in words)

    avg_sentence_length = len(words) / len(sentences) if sentences else 0
    avg_syllables_per_word = total_syllables / len(words) if words else 0

    # Flesch Reading Ease Formula
    score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)

    # Clamp auf [0, 100] und normalisiere auf [0, 1]
    score = max(0, min(100, score))
    return score / 100.0


def flesch_kincaid_grade(text: str) -> float:
    """
    Flesch-Kincaid Grade Level (0-20+, niedriger = leichter).
    Normalisiert auf [0,1]: (20 - grade) / 20 (invertiert, höher = leichter)
    """
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.5

    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.5

    total_syllables = sum(count_syllables(word) for word in words)

    avg_sentence_length = len(words) / len(sentences) if sentences else 0
    avg_syllables_per_word = total_syllables / len(words) if words else 0

    # Flesch-Kincaid Grade Level Formula
    grade = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59

    # Clamp auf [0, 20] und invertiere (höher = leichter)
    grade = max(0, min(20, grade))
    return (20 - grade) / 20.0


def gunning_fog(text: str) -> float:
    """
    Gunning Fog Index (0-20+, niedriger = leichter).
    Normalisiert auf [0,1]: (20 - fog) / 20 (invertiert, höher = leichter)
    """
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.5

    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.5

    # Zähle "complex words" (3+ Silben)
    complex_words = sum(1 for word in words if count_syllables(word) >= 3)
    complex_word_ratio = complex_words / len(words) if words else 0

    avg_sentence_length = len(words) / len(sentences) if sentences else 0

    # Gunning Fog Index Formula
    fog = 0.4 * (avg_sentence_length + 100 * complex_word_ratio)

    # Clamp auf [0, 20] und invertiere (höher = leichter)
    fog = max(0, min(20, fog))
    return (20 - fog) / 20.0


# ---------------------------
# Similarity Baselines (benötigen Referenzen)
# ---------------------------


def compute_rouge(summary: str, reference: str, rouge_type: str = "rougeL") -> float:
    """Berechnet ROUGE-Score (F1)."""
    if not HAS_ROUGE:
        return 0.0
    try:
        scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=False)
        scores = scorer.score(reference, summary)
        return scores[rouge_type].fmeasure  # F1 in [0,1]
    except Exception:
        return 0.0


def compute_bertscore(summary: str, reference: str) -> float:
    """Berechnet BERTScore F1."""
    if not HAS_BERTSCORE:
        return 0.0
    try:
        P, R, F1 = bert_score([summary], [reference], lang="en", verbose=False)
        return float(F1[0].item())
    except Exception:
        return 0.0


def compute_bleu(summary: str, reference: str) -> float:
    """Berechnet BLEU-Score (normalisiert auf [0,1])."""
    if not HAS_BLEU:
        return 0.0
    try:
        bleu = BLEU()
        score = bleu.sentence_score(summary, [reference])
        return score.score / 100.0  # Normalisiere auf [0,1]
    except Exception:
        return 0.0


def compute_meteor(summary: str, reference: str) -> float:
    """Berechnet METEOR-Score."""
    if not HAS_METEOR:
        return 0.0
    try:
        # METEOR benötigt Token-Listen
        summary_tokens = summary.split()
        reference_tokens = reference.split()
        score = meteor_score([reference_tokens], summary_tokens)
        return float(score)
    except Exception:
        return 0.0


# ---------------------------
# Core eval
# ---------------------------


def run_eval(
    rows: list[dict[str, Any]],
    targets: list[str],  # ["readability", "coherence"] oder ["readability"] oder ["coherence"]
    metrics: list[str],  # ["rouge", "bertscore", "flesch", "fk", "fog", "all"]
    max_examples: int | None,
    preds_path: Path,
    gt_min: float,
    gt_max: float,
    seed: int | None,
    bootstrap_n: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Führt die Baseline-Evaluation durch.

    Returns:
        (metrics_dict, predictions_list)
    """
    if seed is not None:
        random.seed(seed)
        random.shuffle(rows)

    # Prüfe, welche Metriken berechnet werden sollen
    compute_rouge_metrics = "rouge" in metrics or "all" in metrics
    compute_bertscore = "bertscore" in metrics or "all" in metrics
    compute_bleu = "bleu" in metrics or "all" in metrics
    compute_meteor = "meteor" in metrics or "all" in metrics
    compute_flesch = "flesch" in metrics or "all" in metrics
    compute_fk = "fk" in metrics or "flesch_kincaid" in metrics or "all" in metrics
    compute_fog = "fog" in metrics or "gunning_fog" in metrics or "all" in metrics

    # Prüfe Referenzen
    has_references = False
    for row in rows[:10]:  # Sample check
        if find_reference(row):
            has_references = True
            break

    if not has_references and (
        compute_rouge_metrics or compute_bertscore or compute_bleu or compute_meteor
    ):
        print(
            "WARNUNG: Keine Referenzen gefunden. ROUGE/BERTScore/BLEU/METEOR werden übersprungen."
        )
        compute_rouge_metrics = False
        compute_bertscore = False
        compute_bleu = False
        compute_meteor = False

    predictions_list: list[dict[str, Any]] = []
    n_seen = 0
    n_used = 0
    n_skipped = 0
    n_no_ref = 0

    if preds_path.exists():
        preds_path.unlink()

    for row in rows:
        if max_examples is not None and n_used >= max_examples:
            break

        n_seen += 1

        summary = row.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            n_skipped += 1
            continue

        meta = row.get("meta", {})
        example_id = meta.get("doc_id") or meta.get("id") or f"example_{n_seen}"

        # Extrahiere GT-Werte
        gt_readability = None
        gt_coherence = None
        if "readability" in targets:
            gt_readability = row.get("gt", {}).get("readability")
        if "coherence" in targets:
            gt_coherence = row.get("gt", {}).get("coherence")

        if not gt_readability and not gt_coherence:
            n_skipped += 1
            continue

        # Normalisiere GT
        gt_readability_norm = None
        gt_coherence_norm = None
        if gt_readability is not None:
            try:
                gt_readability_f = float(gt_readability)
                gt_readability_norm = normalize_to_0_1(gt_readability_f, gt_min, gt_max)
            except Exception:
                pass
        if gt_coherence is not None:
            try:
                gt_coherence_f = float(gt_coherence)
                gt_coherence_norm = normalize_to_0_1(gt_coherence_f, gt_min, gt_max)
            except Exception:
                pass

        if not gt_readability_norm and not gt_coherence_norm:
            n_skipped += 1
            continue

        # Berechne Baseline-Scores
        rec = {
            "example_id": example_id,
            "gt_readability_norm": gt_readability_norm,
            "gt_coherence_norm": gt_coherence_norm,
        }

        # Similarity-Metriken (benötigen Referenz)
        reference = (
            find_reference(row)
            if (compute_rouge_metrics or compute_bertscore or compute_bleu or compute_meteor)
            else None
        )
        if reference:
            if compute_rouge_metrics:
                rec["rouge1"] = compute_rouge(summary, reference, "rouge1")
                rec["rouge2"] = compute_rouge(summary, reference, "rouge2")
                rec["rougeL"] = compute_rouge(summary, reference, "rougeL")
            if compute_bertscore:
                rec["bertscore_f1"] = compute_bertscore(summary, reference)
            if compute_bleu:
                rec["bleu"] = compute_bleu(summary, reference)
            if compute_meteor:
                rec["meteor"] = compute_meteor(summary, reference)
        elif compute_rouge_metrics or compute_bertscore or compute_bleu or compute_meteor:
            n_no_ref += 1

        # Readability-Formeln (brauchen keine Referenz)
        if compute_flesch:
            rec["flesch"] = flesch_reading_ease(summary)
        if compute_fk:
            rec["flesch_kincaid"] = flesch_kincaid_grade(summary)
        if compute_fog:
            rec["gunning_fog"] = gunning_fog(summary)

        predictions_list.append(rec)
        n_used += 1

        with preds_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if n_used % 25 == 0:
            print(f"[{n_used}] (seen={n_seen}, skipped={n_skipped}, no_ref={n_no_ref})")

    if n_no_ref > 0:
        print(f"WARNUNG: {n_no_ref} Beispiele ohne Referenz (Similarity-Metriken nicht berechnet)")

    # Berechne Metriken pro Baseline und Target
    all_metrics = {}

    for target in targets:
        gt_key = f"gt_{target}_norm"
        gt_values = [p.get(gt_key) for p in predictions_list if p.get(gt_key) is not None]

        if not gt_values:
            continue

        for baseline_name in [
            "rouge1",
            "rouge2",
            "rougeL",
            "bertscore_f1",
            "bleu",
            "meteor",
            "flesch",
            "flesch_kincaid",
            "gunning_fog",
        ]:
            baseline_values = [
                p.get(baseline_name)
                for p in predictions_list
                if p.get(baseline_name) is not None and p.get(gt_key) is not None
            ]

            if not baseline_values or len(baseline_values) != len(gt_values):
                continue

            # Berechne Metriken
            pearson_r = pearson(baseline_values, gt_values)
            spearman_rho = spearman(baseline_values, gt_values)
            mae_val = mae(baseline_values, gt_values)
            rmse_val = rmse(baseline_values, gt_values)
            r2_val = r_squared(baseline_values, gt_values)

            # Bootstrap CIs
            pearson_ci = bootstrap_ci(
                pearson, baseline_values, gt_values, n_resamples=bootstrap_n, seed=seed
            )
            spearman_ci = bootstrap_ci(
                spearman, baseline_values, gt_values, n_resamples=bootstrap_n, seed=seed
            )
            mae_ci = bootstrap_ci(
                mae, baseline_values, gt_values, n_resamples=bootstrap_n, seed=seed
            )
            rmse_ci = bootstrap_ci(
                rmse, baseline_values, gt_values, n_resamples=bootstrap_n, seed=seed
            )

            metric_key = f"{baseline_name}_{target}"
            all_metrics[metric_key] = {
                "pearson": {
                    "value": pearson_r,
                    "ci_lower": pearson_ci["ci_lower"],
                    "ci_upper": pearson_ci["ci_upper"],
                },
                "spearman": {
                    "value": spearman_rho,
                    "ci_lower": spearman_ci["ci_lower"],
                    "ci_upper": spearman_ci["ci_upper"],
                },
                "mae": {
                    "value": mae_val,
                    "ci_lower": mae_ci["ci_lower"],
                    "ci_upper": mae_ci["ci_upper"],
                },
                "rmse": {
                    "value": rmse_val,
                    "ci_lower": rmse_ci["ci_lower"],
                    "ci_upper": rmse_ci["ci_upper"],
                },
                "r_squared": r2_val,
                "n": len(baseline_values),
            }

    metrics_dict = {
        "n_seen": n_seen,
        "n_used": n_used,
        "n_skipped": n_skipped,
        "n_no_ref": n_no_ref,
        "targets": targets,
        "metrics_computed": {
            "rouge": compute_rouge_metrics and has_references,
            "bertscore": compute_bertscore and has_references,
            "bleu": compute_bleu and has_references,
            "meteor": compute_meteor and has_references,
            "flesch": compute_flesch,
            "flesch_kincaid": compute_fk,
            "gunning_fog": compute_fog,
        },
        "has_references": has_references,
        "baseline_metrics": all_metrics,
        "gt_normalization": {
            "raw_min": gt_min,
            "raw_max": gt_max,
            "normalized_to": "0..1",
        },
    }

    return metrics_dict, predictions_list


# ---------------------------
# Output generation
# ---------------------------


def write_summary_json(metrics: dict[str, Any], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def write_summary_md(metrics: dict[str, Any], out_path: Path) -> None:
    lines = [
        "# Classical Metrics Baseline Evaluation Summary",
        "",
        "**Dataset:** SummEval",
        f"**Targets:** {', '.join(metrics['targets'])}",
        f"**Examples used:** {metrics['n_used']}",
        f"**Examples skipped:** {metrics['n_skipped']}",
        f"**Examples without reference:** {metrics.get('n_no_ref', 0)}",
        "",
    ]

    if not metrics.get("has_references", False):
        lines.extend(
            [
                "## ⚠️ Keine Referenzen gefunden",
                "",
                "**ROUGE/BERTScore/BLEU/METEOR nicht berechenbar ohne Referenz-Zusammenfassungen.**",
                "",
                "Nur Readability-Formeln (Flesch, Flesch-Kincaid, Gunning Fog) wurden berechnet, da diese keine Referenzen benötigen.",
                "",
                "---",
                "",
            ]
        )

    lines.append("## Baseline Metrics")
    lines.append("")

    baseline_metrics = metrics.get("baseline_metrics", {})
    if not baseline_metrics:
        lines.append("*Keine Metriken berechnet (keine gültigen Baseline-Scores gefunden).*")
    else:
        for metric_key, metric_data in sorted(baseline_metrics.items()):
            baseline_name, target = metric_key.rsplit("_", 1)
            lines.extend(
                [
                    f"### {baseline_name.upper()} vs {target.capitalize()}",
                    "",
                    f"- **Pearson r:** {metric_data['pearson']['value']:.4f} (95% CI: [{metric_data['pearson']['ci_lower']:.4f}, {metric_data['pearson']['ci_upper']:.4f}])",
                    f"- **Spearman ρ:** {metric_data['spearman']['value']:.4f} (95% CI: [{metric_data['spearman']['ci_lower']:.4f}, {metric_data['spearman']['ci_upper']:.4f}])",
                    f"- **MAE:** {metric_data['mae']['value']:.4f} (95% CI: [{metric_data['mae']['ci_lower']:.4f}, {metric_data['mae']['ci_upper']:.4f}])",
                    f"- **RMSE:** {metric_data['rmse']['value']:.4f} (95% CI: [{metric_data['rmse']['ci_lower']:.4f}, {metric_data['rmse']['ci_upper']:.4f}])",
                    f"- **R²:** {metric_data['r_squared']:.4f}",
                    f"- **n:** {metric_data['n']}",
                    "",
                ]
            )

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_run_metadata(
    run_id: str,
    timestamp: str,
    data_path: Path,
    targets: list[str],
    metrics: list[str],
    seed: int | None,
    bootstrap_n: int,
    n_total: int,
    n_used: int,
    n_no_ref: int,
    config_params: dict[str, Any],
    out_path: Path,
) -> None:
    metadata = {
        "run_id": run_id,
        "timestamp": timestamp,
        "git_commit": get_git_commit(),
        "python_version": sys.version,
        "seed": seed,
        "dataset_path": str(data_path),
        "targets": targets,
        "metrics_requested": metrics,
        "n_total": n_total,
        "n_used": n_used,
        "n_no_ref": n_no_ref,
        "config": {
            "bootstrap_n": bootstrap_n,
            **config_params,
        },
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Berechnet klassische Baseline-Metriken für Readability/Coherence"
    )
    ap.add_argument("--data", type=str, required=True, help="Pfad zur JSONL-Datei")
    ap.add_argument(
        "--targets",
        type=str,
        default="readability",
        choices=["readability", "coherence", "both"],
        help="Ziel-Dimension(en)",
    )
    ap.add_argument(
        "--metrics",
        type=str,
        default="all",
        help="Metriken (comma-separated: rouge,bertscore,bleu,meteor,flesch,fk,fog,all)",
    )
    ap.add_argument("--max_examples", type=int, help="Maximale Anzahl Beispiele")
    ap.add_argument("--seed", type=int, help="Random seed für Reproduzierbarkeit")
    ap.add_argument("--bootstrap_n", type=int, default=2000, help="Anzahl Bootstrap-Resamples")
    ap.add_argument(
        "--out_dir", type=str, help="Output-Verzeichnis (default: results/evaluation/baselines)"
    )
    ap.add_argument("--gt-min", type=float, default=1.0, help="GT-Minimum (default: 1.0)")
    ap.add_argument("--gt-max", type=float, default=5.0, help="GT-Maximum (default: 5.0)")

    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        ap.error(f"Datei nicht gefunden: {data_path}")

    # Parse targets
    if args.targets == "both":
        targets = ["readability", "coherence"]
    else:
        targets = [args.targets]

    # Parse metrics
    metrics_str = args.metrics.lower()
    if metrics_str == "all":
        metrics = ["all"]
    else:
        metrics = [m.strip() for m in metrics_str.split(",")]

    max_examples = args.max_examples
    seed = args.seed
    bootstrap_n = args.bootstrap_n

    rows = load_jsonl(data_path)

    # Run ID
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    targets_str = "_".join(targets)
    metrics_str_short = "_".join(metrics[:3])  # Erste 3 Metriken für Run-ID
    run_id = f"baselines_{targets_str}_{metrics_str_short}_{ts}"
    if seed is not None:
        run_id += f"_seed{seed}"

    # Output directory
    if args.out_dir:
        out_dir = Path(args.out_dir) / run_id
    else:
        out_dir = Path("results") / "evaluation" / "baselines" / run_id
    ensure_dir(out_dir)

    preds_path = out_dir / "predictions.jsonl"
    summary_json_path = out_dir / "summary.json"
    summary_md_path = out_dir / "summary.md"
    metadata_path = out_dir / "run_metadata.json"

    print(f"Data: {data_path} (rows={len(rows)})")
    print(f"Targets: {targets}")
    print(f"Metrics: {metrics}")
    print(f"GT scale: [{args.gt_min}, {args.gt_max}] -> [0,1]")
    print(f"Seed: {seed}")
    print(f"Bootstrap resamples: {bootstrap_n}")
    print(f"Output: {out_dir}")

    metrics_dict, predictions_list = run_eval(
        rows=rows,
        targets=targets,
        metrics=metrics,
        max_examples=max_examples,
        preds_path=preds_path,
        gt_min=args.gt_min,
        gt_max=args.gt_max,
        seed=seed,
        bootstrap_n=bootstrap_n,
    )

    # Write outputs
    write_summary_json(metrics_dict, summary_json_path)
    write_summary_md(metrics_dict, summary_md_path)
    write_run_metadata(
        run_id=run_id,
        timestamp=ts,
        data_path=data_path,
        targets=targets,
        metrics=metrics,
        seed=seed,
        bootstrap_n=bootstrap_n,
        n_total=len(rows),
        n_used=metrics_dict["n_used"],
        n_no_ref=metrics_dict.get("n_no_ref", 0),
        config_params={
            "max_examples": max_examples,
            "gt_min": args.gt_min,
            "gt_max": args.gt_max,
        },
        out_path=metadata_path,
    )

    print("\n" + "=" * 60)
    print("Baseline-Evaluation abgeschlossen!")
    print("=" * 60)
    print(f"\nArtefakte gespeichert in: {out_dir}")


if __name__ == "__main__":
    main()
