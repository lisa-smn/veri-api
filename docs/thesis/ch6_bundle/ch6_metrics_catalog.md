# Evaluationsmetriken – Metrik-Katalog

**Erstellt:** 2026-01-30  
**Quelle:** Code-Analyse aller Evaluations-Scripts

---

## A) Metrik-Katalog (Tabelle)

| Dimension | Metrikname | Definition kurz | Implementierung (Library/Function) | Input (Labels/Scores) | Aggregation | Output-Datei | Code-Referenz |
|---|---|---|---|---|---|---|---|
| **Factuality (Klassifikation)** |
| Factuality | TP, FP, TN, FN | Confusion Matrix Zellen | Custom (BinaryMetrics) | pred_has_error (bool), gt_has_error (bool) | Pro Run (summiert über alle Beispiele) | summary.json: `confusion_matrix` | `app/services/analysis/metrics.py:13-16`<br>`scripts/run_m10_factuality.py:522-530` |
| Factuality | Accuracy | (TP + TN) / (TP + FP + TN + FN) | Custom (BinaryMetrics.accuracy) | TP, FP, TN, FN | Pro Run | summary.json: `metrics.accuracy` | `app/services/analysis/metrics.py:19-21` |
| Factuality | Precision | TP / (TP + FP) | Custom (BinaryMetrics.precision) | TP, FP | Pro Run | summary.json: `metrics.precision` | `app/services/analysis/metrics.py:24-25` |
| Factuality | Recall | TP / (TP + FN) | Custom (BinaryMetrics.recall) | TP, FN | Pro Run | summary.json: `metrics.recall` | `app/services/analysis/metrics.py:28-29` |
| Factuality | F1 | 2 * (Precision * Recall) / (Precision + Recall) | Custom (BinaryMetrics.f1) | Precision, Recall | Pro Run | summary.json: `metrics.f1` | `app/services/analysis/metrics.py:32-34` |
| Factuality | Specificity | TN / (TN + FP) | Custom (BinaryMetrics.specificity) | TN, FP | Pro Run | summary.json: `metrics.specificity` | `app/services/analysis/metrics.py:37-38` |
| Factuality | Balanced Accuracy | (Recall + Specificity) / 2 | Custom (berechnet) | Recall, Specificity | Pro Run | summary.json: `metrics.balanced_accuracy` | `scripts/run_m10_factuality.py:540`<br>`scripts/eval_frank_factuality_agent_on_manifest.py:146-147` |
| Factuality | MCC | (TP×TN - FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN)) | Custom (compute_mcc) | TP, FP, TN, FN | Pro Run | summary.json: `metrics.mcc` | `scripts/eval_frank_factuality_agent_on_manifest.py:155-159` |
| Factuality | AUROC | Area Under ROC Curve | Custom (compute_auroc) | agent_score (float [0,1]), gt_has_error (bool) | Pro Run | summary.json: `metrics.auroc` | `app/services/analysis/metrics.py:54-88`<br>`scripts/eval_frank_factuality_agent_on_manifest.py:204-238` |
| **Coherence/Readability (Regression/Korrelation)** |
| Coherence/Readability | Pearson r | Lineare Korrelation zwischen pred und gt | Custom (pearson) | pred (float [0,1]), gt_norm (float [0,1]) | Pro Run (über alle Beispiele) | summary.json: `pearson.value` | `scripts/eval_sumeval_coherence.py:103-112` |
| Coherence/Readability | Spearman ρ | Rangkorrelation zwischen pred und gt | Custom (spearman via pearson auf Rängen) | pred (float [0,1]), gt_norm (float [0,1]) | Pro Run | summary.json: `spearman.value` | `scripts/eval_sumeval_coherence.py:131-134` |
| Coherence/Readability | MAE | Mean Absolute Error | Custom (mae) | pred (float [0,1]), gt_norm (float [0,1]) | Pro Run (Mittelwert) | summary.json: `mae.value` | `scripts/eval_sumeval_coherence.py:137-140` |
| Coherence/Readability | RMSE | Root Mean Squared Error | Custom (rmse) | pred (float [0,1]), gt_norm (float [0,1]) | Pro Run (Mittelwert) | summary.json: `rmse.value` | `scripts/eval_sumeval_coherence.py:143-146` |
| Coherence/Readability | R² | Coefficient of Determination | Custom (r_squared) | pred (float [0,1]), gt_norm (float [0,1]) | Pro Run | summary.json: `r_squared` | `scripts/eval_sumeval_coherence.py:149-158` |
| **Baselines** |
| Baselines | ROUGE-L | Longest Common Subsequence F1 | rouge-score (RougeScorer) | summary (str), reference (str) | Pro Beispiel, dann Korrelation mit GT | predictions.jsonl: `rouge_l` | `scripts/eval_sumeval_coherence_baselines.py:229-235`<br>`scripts/eval_frank_factuality_baselines.py:476-484` |
| Baselines | BERTScore | Contextual Embedding Similarity F1 | bert-score (bert_score) | summary (str), reference (str) | Pro Beispiel, dann Korrelation mit GT | predictions.jsonl: `bertscore_f1` | `scripts/eval_sumeval_coherence_baselines.py:238-248`<br>`scripts/eval_frank_factuality_baselines.py:487-499` |
| **Bootstrap/Unsicherheit** |
| Alle | Bootstrap CI (95%) | 2.5th und 97.5th Percentile nach n_resamples | Custom (bootstrap_ci / bootstrap_ci_binary) | Metrik-Funktion, Daten | Pro Metrik | summary.json: `*.ci_lower`, `*.ci_upper` | `scripts/eval_sumeval_coherence.py:166-208`<br>`scripts/eval_frank_factuality_agent_on_manifest.py:162-201` |

---

## B) Skalierung und Normalisierung

### Ground Truth Normalisierung (Coherence/Readability)

**Formel:** `gt_norm = (gt_raw - 1) / 4`

- **Input:** Raw GT-Skala [1.0, 5.0] (aus SummEval Human Ratings)
- **Output:** Normalized GT [0.0, 1.0]
- **Code:** `scripts/eval_sumeval_coherence.py:14-17`
- **Konfiguration:** `gt_min=1.0`, `gt_max=5.0` (default, konfigurierbar)

### Predictions

- **Agent-Scores:** Bereits in [0, 1] (aus Agent-Output)
- **Baseline-Scores:** ROUGE-L und BERTScore bereits in [0, 1] (F1-Scores)

### Schwellenwerte (Factuality)

- **Decision Threshold:** `error_threshold=1` (Anzahl Issues, default)
- **Score Cutoff:** Optional `decision_threshold_float` (für gewichtete Aggregation)
- **Code:** `scripts/run_m10_factuality.py:265-474`

---

## C) Bootstrap-Resampling

### Parameter

- **n_resamples:** 2000 (default)
- **confidence:** 0.95 (95% CI)
- **seed:** Konfigurierbar (z.B. 42)
- **Method:** Percentile-Methode (2.5th und 97.5th Percentile)

### Implementierung

**Für Regression-Metriken (Pearson, Spearman, MAE, RMSE):**
- `scripts/eval_sumeval_coherence.py:166-208`
- Resampling mit Replacement, dann Metrik-Funktion auf Resample anwenden

**Für Binäre Metriken (Accuracy, Precision, Recall, F1, Balanced Accuracy):**
- `scripts/eval_frank_factuality_agent_on_manifest.py:162-201`
- Resampling mit Replacement, dann Confusion Matrix berechnen, dann Metrik

### Gebootstrappte Metriken

**Coherence/Readability:**
- Pearson r (CI)
- Spearman ρ (CI)
- MAE (CI)
- RMSE (CI)

**Factuality:**
- Accuracy (CI)
- Precision (CI)
- Recall (CI)
- F1 (CI)
- Balanced Accuracy (CI)
- MCC (CI, optional)

**Nicht gebootstrappt:**
- R² (nur Wert, kein CI)
- AUROC (nur Wert, kein CI)
- Specificity (nur Wert, kein CI)

---

## D) Output-Formate

### summary.json

**Struktur:**
```json
{
  "n_used": 200,
  "n_failed": 0,
  "pearson": {
    "value": 0.371,
    "ci_lower": 0.197,
    "ci_upper": 0.557
  },
  "spearman": { ... },
  "mae": { ... },
  "rmse": { ... },
  "r_squared": 0.075,
  "gt_normalization": {
    "raw_min": 1.0,
    "raw_max": 5.0,
    "normalized_to": "0..1"
  }
}
```

**Oder für Factuality:**
```json
{
  "n_used": 200,
  "confusion_matrix": {
    "tp": 99,
    "fp": 27,
    "tn": 49,
    "fn": 25
  },
  "metrics": {
    "f1": {
      "value": 0.79,
      "ci_lower": 0.73,
      "ci_upper": 0.84
    },
    "precision": { ... },
    "recall": { ... },
    "balanced_accuracy": { ... },
    "auroc": 0.89,
    "mcc": { ... }
  }
}
```

**Code:** `scripts/eval_sumeval_coherence.py:605-610` (Regression)<br>`scripts/eval_frank_factuality_agent_on_manifest.py:345-354` (Klassifikation)

### summary.md

Human-readable Markdown-Format mit formatierten Metriken und CIs.

**Code:** `scripts/eval_sumeval_coherence.py:611-651`

### predictions.jsonl

Eine JSON-Zeile pro Beispiel mit:
- `example_id`
- `pred` / `pred_agent` (float [0,1])
- `gt_norm` (float [0,1])
- `gt_raw` (float [1,5])
- `issue_spans` (optional, für Factuality)
- `baseline_scores` (optional, für Baselines)

**Code:** `scripts/eval_sumeval_coherence.py:543` (Schreiben)

---

## E) Diskrepanzen (Code vs. Dokumentation)

**Keine gefunden:** Alle in der Dokumentation erwähnten Metriken werden auch im Code berechnet.

**Hinweis:** R² wird berechnet, aber nicht immer in allen Reports angezeigt (abhängig vom Script).

---

## F) Aggregationslogik

### Pro-Beispiel → Pro-Run

**Regression-Metriken:**
- Alle Metriken werden über alle Beispiele aggregiert (Pearson, Spearman, MAE, RMSE, R²)

**Klassifikationsmetriken:**
- Confusion Matrix wird über alle Beispiele summiert (TP, FP, TN, FN)
- Accuracy, Precision, Recall, F1, Specificity werden aus aggregierter Confusion Matrix berechnet

**Baselines:**
- ROUGE-L / BERTScore werden pro Beispiel berechnet
- Dann Korrelation (Pearson, Spearman) mit GT über alle Beispiele

### Keine Macro/Micro/Weighted Aggregation

- Alle Metriken sind "macro" (ein Wert pro Run, nicht pro Klasse)
- Keine gewichtete Aggregation nach Klassenverteilung
