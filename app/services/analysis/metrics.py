"""
Metrik-Berechnung für Evaluation.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class BinaryMetrics:
    """Metriken für binäre Klassifikation."""

    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0.0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def specificity(self) -> float:
        return self.tn / (self.tn + self.fp) if (self.tn + self.fp) > 0 else 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "specificity": self.specificity,
        }


def compute_auroc(predictions: list[float], ground_truths: list[bool]) -> float:
    """Berechnet AUROC (vereinfacht, für echte Implementierung scipy.stats verwenden)."""
    if not predictions or not ground_truths:
        return 0.0

    # Sort by prediction score (descending)
    pairs = list(zip(predictions, ground_truths))
    pairs.sort(key=lambda x: x[0], reverse=True)

    # Count positives and negatives
    num_positives = sum(1 for _, gt in pairs if gt)
    num_negatives = len(pairs) - num_positives

    if num_positives == 0 or num_negatives == 0:
        return 0.5  # No discrimination possible

    # Calculate AUC using trapezoidal rule
    tpr = 0.0
    fpr = 0.0
    auc = 0.0
    prev_tpr = 0.0
    prev_fpr = 0.0

    for pred, gt in pairs:
        if gt:
            tpr += 1.0 / num_positives
        else:
            fpr += 1.0 / num_negatives

        # Add area of trapezoid
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_tpr = tpr
        prev_fpr = fpr

    return auc


def compute_threshold_sweep(
    predictions: list[float],
    ground_truths: list[bool],
    thresholds: list[float],
) -> list[dict[str, float]]:
    """Sweep über Thresholds und berechnet Metriken."""
    results = []
    for threshold in thresholds:
        pred_binary = [p >= threshold for p in predictions]
        metrics = BinaryMetrics()
        for pred, gt in zip(pred_binary, ground_truths):
            if pred and gt:
                metrics.tp += 1
            elif pred and not gt:
                metrics.fp += 1
            elif not pred and not gt:
                metrics.tn += 1
            else:
                metrics.fn += 1

        results.append(
            {
                "threshold": threshold,
                **metrics.to_dict(),
            }
        )
    return results


def analyze_error_patterns(
    examples: list[dict[str, Any]],
    predictions: list[bool],
    ground_truths: list[bool],
) -> dict[str, Any]:
    """Analysiert Fehlermuster (FP/FN)."""
    fp_examples = []
    fn_examples = []

    for ex, pred, gt in zip(examples, predictions, ground_truths):
        if pred and not gt:
            fp_examples.append(ex)
        elif not pred and gt:
            fn_examples.append(ex)

    # Analyze common patterns
    fp_issue_types = {}
    fn_issue_types = {}

    for ex in fp_examples:
        issue_spans = ex.get("issue_spans", [])
        for span in issue_spans:
            issue_type = span.get("issue_type", "OTHER")
            fp_issue_types[issue_type] = fp_issue_types.get(issue_type, 0) + 1

    for ex in fn_examples:
        issue_spans = ex.get("issue_spans", [])
        for span in issue_spans:
            issue_type = span.get("issue_type", "OTHER")
            fn_issue_types[issue_type] = fn_issue_types.get(issue_type, 0) + 1

    return {
        "num_fp": len(fp_examples),
        "num_fn": len(fn_examples),
        "fp_issue_types": fp_issue_types,
        "fn_issue_types": fn_issue_types,
        "fp_examples": fp_examples[:10],  # Top 10
        "fn_examples": fn_examples[:10],
    }


def analyze_subsets(
    examples: list[dict[str, Any]],
    predictions: list[bool],
    ground_truths: list[bool],
    subset_key: str = "meta",
) -> dict[str, dict[str, float]]:
    """Analysiert Metriken pro Subset."""
    subsets = {}

    for ex, pred, gt in zip(examples, predictions, ground_truths):
        meta = ex.get(subset_key, {})
        if isinstance(meta, dict):
            # Use first key as subset identifier
            subset_id = str(list(meta.values())[0]) if meta else "unknown"
        else:
            subset_id = str(meta) if meta else "unknown"

        if subset_id not in subsets:
            subsets[subset_id] = BinaryMetrics()

        metrics = subsets[subset_id]
        if pred and gt:
            metrics.tp += 1
        elif pred and not gt:
            metrics.fp += 1
        elif not pred and not gt:
            metrics.tn += 1
        else:
            metrics.fn += 1

    return {k: v.to_dict() for k, v in subsets.items()}
