# Chapter 6 – Quantitative Results Tables

**Quelle:** `results/evaluation/**/summary.json` und `results/evaluation/summary.md`

---

## Factuality (FRANK, n=50)

**Run:** `evidence_gate_test_count_as_error`  
**Run-Manifest:** `results/evaluation/runs/results/evidence_gate_test_count_as_error.json`  
**Summary:** `results/evaluation/summary.md`

| Dataset | Run ID | N | PosRate | TP | FP | TN | FN | Precision | Recall | F1 | Specificity | Accuracy | AUROC | BalancedAcc | Quelle |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| frank | evidence_gate_test_count_as_error | 50 | 0.940 | 38 | 2 | 1 | 9 | 0.950 | 0.809 | 0.874 | 0.333 | 0.780 | 0.787 | 0.571 | `results/evaluation/runs/results/evidence_gate_test_count_as_error.json:4-14` |

**Labelverteilung:**
- Pos (has_error=true): 47
- Neg (has_error=false): 3
- PosRate: 0.940

**Quelle:** `results/evaluation/runs/results/evidence_gate_test_count_as_error.json:16-19`

---

## Coherence (SummEval, n=200)

**Run:** `coherence_20260107_205123_gpt-4o-mini_v1_seed42`  
**Run-Ordner:** `results/evaluation/coherence/coherence_20260107_205123_gpt-4o-mini_v1_seed42/`  
**Summary:** `results/evaluation/coherence/coherence_20260107_205123_gpt-4o-mini_v1_seed42/summary.json`

| Method | n | Spearman ρ (95% CI) | Pearson r (95% CI) | MAE (95% CI) | RMSE (95% CI) | R² | Quelle |
|---|---:|---|---|---|---|---:|---|
| Coherence-Agent | 200 | 0.451 [0.314, 0.575] | 0.371 [0.197, 0.557] | 0.174 [0.152, 0.197] | 0.237 [0.203, 0.273] | 0.075 | `results/evaluation/coherence/coherence_20260107_205123_gpt-4o-mini_v1_seed42/summary.json:11-26` |

**Details:**
- `n_used`: 200
- `n_failed`: 0
- `gt_normalization`: raw_min=1.0, raw_max=5.0, normalized_to="0..1"

**Quelle:** `results/evaluation/coherence/coherence_20260107_205123_gpt-4o-mini_v1_seed42/summary.json:2-31`

**Hinweis:** LLM-Judge und Baselines sind Doc-Evidence (vgl. `docs/status_pack/2026-01-08/03_evaluation_results.md:27-28`)

---

## Readability (SummEval, n=200)

**Run:** `readability_20260116_170832_gpt-4o-mini_v1_seed42`  
**Run-Ordner:** `results/evaluation/readability/readability_20260116_170832_gpt-4o-mini_v1_seed42/`  
**Summary:** `results/evaluation/readability/readability_20260116_170832_gpt-4o-mini_v1_seed42/summary.json`

| Method | n | Spearman ρ (95% CI) | Pearson r (95% CI) | MAE (95% CI) | RMSE (95% CI) | R² | Quelle |
|---|---:|---|---|---|---|---:|---|
| Readability-Agent | 200 | 0.424 [0.297, 0.532] | 0.401 [0.312, 0.478] | 0.285 [0.265, 0.305] | 0.318 [0.303, 0.334] | -2.824 | `results/evaluation/readability/readability_20260116_170832_gpt-4o-mini_v1_seed42/summary.json:11-26` |

**Details:**
- `n_used`: 200
- `n_failed`: 0
- `gt_normalization`: raw_min=1.0, raw_max=5.0, normalized_to="0..1"
- `collapse_detected`: false
- `cache_stats`: cache_mode="write", cache_hits=12, cache_misses=188

**Quelle:** `results/evaluation/readability/readability_20260116_170832_gpt-4o-mini_v1_seed42/summary.json:2-39`

**Hinweis:** LLM-Judge und Baselines sind Doc-Evidence (vgl. `docs/status_pack/2026-01-08/03_evaluation_results.md:109-112`)

---

## Factuality (FRANK Manifest, n=200) – Doc-Evidence

**Run:** `factuality_agent_manifest_20260107_215431_gpt-4o-mini`  
**Run-Ordner:** `results/evaluation/factuality/factuality_agent_manifest_20260107_215431_gpt-4o-mini/`  
**Summary:** `results/evaluation/factuality/factuality_agent_manifest_20260107_215431_gpt-4o-mini/summary.json`

| Metric | Value | 95% CI | Quelle |
|---|---:|---|---|
| F1 | 0.79 | [0.73, 0.84] | `results/evaluation/factuality/factuality_agent_manifest_20260107_215431_gpt-4o-mini/summary.json:13-17` |
| Precision | 0.79 | [0.71, 0.86] | `results/evaluation/factuality/factuality_agent_manifest_20260107_215431_gpt-4o-mini/summary.json:29-33` |
| Recall | 0.80 | [0.73, 0.87] | `results/evaluation/factuality/factuality_agent_manifest_20260107_215431_gpt-4o-mini/summary.json:34-38` |
| AUROC | 0.89 | - | `results/evaluation/factuality/factuality_agent_manifest_20260107_215431_gpt-4o-mini/summary.json:23` |
| Balanced Accuracy | 0.72 | [0.66, 0.79] | `results/evaluation/factuality/factuality_agent_manifest_20260107_215431_gpt-4o-mini/summary.json:18-22` |
| MCC | 0.45 | [0.31, 0.57] | `results/evaluation/factuality/factuality_agent_manifest_20260107_215431_gpt-4o-mini/summary.json:24-28` |
| Accuracy | 0.74 | [0.68, 0.80] | `results/evaluation/factuality/factuality_agent_manifest_20260107_215431_gpt-4o-mini/summary.json:39-43` |
| Specificity | 0.64 | - | `results/evaluation/factuality/factuality_agent_manifest_20260107_215431_gpt-4o-mini/summary.json:44` |

**Confusion Matrix:**
- TP: 99
- FP: 27
- TN: 49
- FN: 25

**Quelle:** `results/evaluation/factuality/factuality_agent_manifest_20260107_215431_gpt-4o-mini/summary.json:6-11`

**Hinweis:** `predictions.jsonl` fehlt; Metriken aus Status-Pack übernommen (vgl. `docs/status_pack/2026-01-08/03_evaluation_results.md:59-65`)

---

## Notes

1. **Bootstrap Confidence Intervals:** Alle CIs wurden mit Bootstrap-Resampling (n=2000) berechnet (vgl. `scripts/eval_sumeval_coherence.py`, `scripts/eval_sumeval_readability.py`).

2. **Doc-Evidence:** FRANK Manifest Metriken sind Doc-Evidence, da `predictions.jsonl` fehlt. Coherence/Readability sind vollständig als Repo-Evidence verfügbar.

3. **R² Interpretation:** Negative R² Werte (z.B. Readability: -2.824) deuten darauf hin, dass das Modell schlechter als ein Null-Modell (Mittelwert) abschneidet. Dies kann bei nicht-linearen Beziehungen oder hoher Varianz auftreten.
