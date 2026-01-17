# Readability Evaluation Results

**Datensatz:** SummEval | **n:** 200 | **Seed:** 42  
**Label:** `gt.readability` (1-5) → normalisiert auf [0,1] via `(gt_raw - 1) / 4`

---

## Metriken (v1 raw - Hauptresultat)

| Metric | Value | 95% CI |
|--------|-------|--------|
| Spearman ρ | 0.4053 | [0.2743, 0.5169] |
| Pearson r | 0.3865 | [0.2946, 0.4613] |
| MAE | 0.2845 | [0.2643, 0.3042] |
| RMSE | 0.3193 | [0.3030, 0.3352] |
| R² | -2.84 | - |

---

## Interpretation

**v1 raw zeigt moderate Ranking-Korrelation (Spearman ρ = 0.41)** mit menschlichen Ratings. MAE = 0.28 bedeutet durchschnittliche Abweichung von 0.28 Punkten (auf Skala 0-1). Unsicherheit via Bootstrapping zeigt konsistent positive Korrelation (alle CIs > 0). R² = -2.84 deutet auf Kalibrierungsproblem hin, aber **für Ranking-basierte Evaluation ist v1 raw die beste Wahl**.

**Calibration Trade-off:** Kalibrierung kann MAE senken (z.B. v1 linear-cal: MAE = 0.14), aber verschlechtert das Ranking (Spearman sinkt auf 0.16). **Empfehlung:** v1 raw für Ranking, Kalibrierung nur wenn absolute Werte wichtiger sind als Ranking.

---

**Run:** `results/evaluation/readability/readability_20260114_184754_gpt-4o-mini_v1_seed42/`  
**Matrix:** `results/evaluation/readability/summary_matrix.csv/.md`  
**Improvement Analysis:** `docs/status_pack/2026-01-08/readability_improvement_attempt.md`  
**Error Analysis:** `results/evaluation/readability/analysis_readability_v1.md`
