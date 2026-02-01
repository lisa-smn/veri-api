# Readability Evaluation Summary

**Dataset:** SummEval
**Examples used:** 200
**Examples skipped:** 0
**Examples failed:** 0

## Cache

- **Cache Mode:** write
- **Cache Hits:** 12
- **Cache Misses:** 188

## Setup

- **Label-Definition:** `gt.readability` (1-5 Likert-Skala)
- **Normalisierung:** `gt_norm = (gt_raw - 1.0) / (5.0 - 1.0)`

## Metrics

### Correlation

- **Pearson r:** 0.4146 (95% CI: [0.3128, 0.4995])
- **Spearman ρ:** 0.4201 (95% CI: [0.2929, 0.5322])

### Error Metrics

- **MAE:** 0.2707 (95% CI: [0.2522, 0.2891])
- **RMSE:** 0.3030 (95% CI: [0.2868, 0.3184])
- **R²:** -2.4613

### Ground Truth Normalization

- Raw scale: [1.0, 5.0]
- Normalized to: 0..1

## Distributions

### GT (gt_norm) Distribution

- [0.0, 0.2): 2
- [0.2, 0.4): 2
- [0.4, 0.6): 12
- [0.6, 0.8): 8
- [0.8, 1.0): 32
- [1.0, 1.0]: 144

### Predictions (pred) Distribution

Prediction distribution (counts per bucket/value):

- [0.4, 0.6): 36
- [0.6, 0.8): 104
- [0.8, 1.0): 53
- [1.0, 1.0]: 7

---

## Qualitative Examples (Top 5 High-Severity Issues)

### Example 1
- **ID:** dm-test-8764fb95bfad8ee849274873a92fb8d6b400eee2_7a70cec5
- **GT:** 3.33 (norm: 0.583)
- **Pred:** 0.600
- **Issues:** 2 (max severity: high)
- **Top Issues:** None (high), None (medium)

### Example 2
- **ID:** dm-test-8764fb95bfad8ee849274873a92fb8d6b400eee2_852bfe84
- **GT:** 5.00 (norm: 1.000)
- **Pred:** 0.600
- **Issues:** 2 (max severity: high)
- **Top Issues:** None (high), None (medium)

### Example 3
- **ID:** dm-test-f26d8400ae49b90d109c165d0f44b8f6ca253c08_a9b53419
- **GT:** 5.00 (norm: 1.000)
- **Pred:** 0.600
- **Issues:** 2 (max severity: high)
- **Top Issues:** None (high), None (medium)

### Example 4
- **ID:** dm-test-f26d8400ae49b90d109c165d0f44b8f6ca253c08_f78909af
- **GT:** 5.00 (norm: 1.000)
- **Pred:** 0.600
- **Issues:** 2 (max severity: high)
- **Top Issues:** None (high), None (medium)

### Example 5
- **ID:** dm-test-f26d8400ae49b90d109c165d0f44b8f6ca253c08_38101391
- **GT:** 5.00 (norm: 1.000)
- **Pred:** 0.600
- **Issues:** 2 (max severity: high)
- **Top Issues:** None (high), None (medium)

## Limitations

- SummEval Readability-Ratings haben intrinsische Varianz (Rater-Noise), was die Unsicherheit erhöht.
- Die Bootstrap-CIs quantifizieren diese Unsicherheit, aber größere Samples würden engere CIs liefern.