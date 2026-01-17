# Readability Improvement Attempt

**Datum:** 2026-01-15

## Vergleichstabelle (aus summary_matrix)

| Version | Pearson r (95% CI) | Spearman ρ (95% CI) | MAE (95% CI) | RMSE (95% CI) | R² | n |
|---------|-------------------|---------------------|--------------|---------------|-----|---|
| **v1 raw** | 0.3865 [0.2946, 0.4613] | 0.4053 [0.2743, 0.5169] | 0.2845 [0.2643, 0.3042] | 0.3193 [0.3030, 0.3352] | -2.8426 | 200 |
| **v1 linear-cal** | 0.1176 [-0.0150, 0.2431] | 0.1566 [0.0216, 0.2852] | 0.1449 [0.1314, 0.1589] | 0.1749 [0.1514, 0.1974] | -0.1531 | 200 |
| **v1 isotonic-cal** | 0.1111 [-0.0293, 0.2485] | 0.1566 [0.0216, 0.2852] | 0.2277 [0.2040, 0.2521] | 0.2858 [0.2585, 0.3112] | -2.0799 | 200 |
| **v2 raw** | 0.0394 [-0.0819, 0.1260] | 0.0330 [-0.0912, 0.1503] | 0.6554 [0.6317, 0.6787] | 0.6767 [0.6581, 0.6943] | -16.2605 | 200 |
| **v2 calibrated** | *Ausstehend* | *Ausstehend* | *Ausstehend* | *Ausstehend* | *Ausstehend* | *Ausstehend* |

### Prediction Distribution (counts per bucket/value)

**v1 raw:**
- [0.4, 0.6): 41
- [0.6, 0.8): 105
- [0.8, 1.0): 47
- [1.0, 1.0]: 7

**v1 linear-cal (⚠️ COLLAPSE):**
- [0.8, 1.0): 200 (100% im selben Bucket)

**v1 isotonic-cal:**
- [0.4, 0.6): 51
- [0.8, 1.0): 149

**v2 raw (⚠️ COLLAPSE):**
- [0.2, 0.4): 183 (91.5% im selben Bucket)
- [0.4, 0.6): 17

---

## Interpretation

### 1. v1 raw best ranking

**v1 raw zeigt die beste Ranking-Korrelation:**
- **Spearman ρ = 0.41** (moderate Korrelation, CI schließt 0 nicht ein)
- **MAE = 0.28** (akzeptabel für 0-1 Skala)
- **R² = -2.84** (negativ, aber deutlich besser als v2)
- **Verteilung:** Ausgewogen über [0.4, 1.0], keine Kollaps

**Empfehlung:** v1 raw ist die Basis für Ranking-basierte Evaluation.

### 2. v2 output collapse (konstant ~0.25)

**v2 zeigt deutlichen Output-Collapse:**
- **Spearman ρ = 0.033** (praktisch keine Korrelation, CI schließt 0 ein)
- **MAE = 0.66** (sehr hoch)
- **R² = -16.26** (extrem negativ)
- **Verteilung:** 91.5% der Predictions in [0.2, 0.4) → Kollaps

**Ursache:** Prompt v2 (1-5 integer score) führt zu eingeschränkter LLM-Ausgabe. Das LLM gibt fast nur den Wert 2 (entspricht 0.25 nach Normalisierung) aus.

### 3. Calibration kann MAE senken ohne Ranking zu verbessern (Warnung vor MAE-only)

**v1 linear-cal zeigt das Problem:**
- **MAE = 0.14** (deutlich besser als v1 raw: 0.28)
- **Spearman ρ = 0.16** (schlechter als v1 raw: 0.41)
- **R² = -0.15** (besser als v1 raw: -2.84)
- **⚠️ COLLAPSE:** 100% der Predictions in [0.8, 1.0)

**Lektion:** MAE allein ist nicht ausreichend - Spearman zeigt die echte Qualität des Rankings. Kalibrierung kann die Skala korrigieren (MAE sinkt), aber wenn das Ranking schlecht ist, bleibt Spearman niedrig.

### 4. Linear vs Isotonic (kein isotonic Vorteil)

**v1 isotonic-cal vs v1 linear-cal:**
- **Spearman:** Beide 0.16 (gleich)
- **MAE:** Isotonic 0.23 vs Linear 0.14 (Linear besser)
- **R²:** Isotonic -2.08 vs Linear -0.15 (Linear besser)
- **Verteilung:** Isotonic weniger kollabiert (51 in [0.4, 0.6), 149 in [0.8, 1.0))

**Fazit:** Isotonic zeigt keinen klaren Vorteil. Beide Kalibrierungsmethoden verschlechtern das Ranking (Spearman sinkt von 0.41 auf 0.16).

### 5. Empfehlung: v1 raw fürs Ranking, calibration optional nur für skalennahe Ausgabe

**Für Ranking-basierte Evaluation:**
- **v1 raw** ist die beste Wahl (Spearman ρ = 0.41)
- Keine Kalibrierung verwenden, da sie das Ranking verschlechtert

**Für skalennahe Ausgabe (wenn absolute Werte wichtig sind):**
- Kalibrierung kann MAE senken, aber:
  - Ranking verschlechtert sich (Spearman sinkt)
  - Risiko von Output-Collapse (wie bei v1 linear-cal)
- **Empfehlung:** Nur verwenden, wenn absolute Werte wichtiger sind als Ranking

### 6. Next Steps (abhängig von Mechanismus)

**Mechanismus:** LLM-judge (primär), siehe `readability_score_mechanism.md`

**Empfohlene Verbesserungen:**
1. **Prompt-Optimierung:** v1 beibehalten, v2 verwerfen (Output-Collapse)
2. **Robustes Parsing:** JSON-Parsing mit Retry-Logik (bereits implementiert)
3. **Collapse-Detection:** Automatische Warnung bei >80% im selben Bucket (bereits implementiert)
4. **Keine Kalibrierung für Ranking:** Nur für absolute Werte, wenn nötig

---

## Run-Ordner

- **v1 raw:** `results/evaluation/readability/readability_20260114_184754_gpt-4o-mini_v1_seed42/`
- **v1 linear-cal:** `results/evaluation/readability/readability_20260114_222224_gpt-4o-mini_v1_seed42/`
- **v1 isotonic-cal:** `results/evaluation/readability/readability_20260115_193314_gpt-4o-mini_v1_seed42/`
- **v2 raw:** `results/evaluation/readability/readability_20260114_201157_gpt-4o-mini_v2_seed42/`

---

## Error Analysis

**v1 raw Error Analysis:**
- `results/evaluation/readability/analysis_readability_v1.md`
- Analysiert 200 Summaries mit größten Fehlern, GT/Pred-Diskrepanzen, High-Severity-Issues

---

## Matrix-Dateien

- **CSV:** `results/evaluation/readability/summary_matrix.csv`
- **MD:** `results/evaluation/readability/summary_matrix.md`

---

## Kalibrierungsdetails

**Clipping-Verhalten:**
- Bei linearer Kalibrierung: `pred_cal = a * pred + b`, dann **geclippt auf [0,1]** (keine Rundung auf Buckets)
- Bei isotonischer Kalibrierung: Monotone Funktion, dann **geclippt auf [0,1]** (keine Rundung auf Buckets)
- **Wichtig:** Keine Diskretisierung/Bucketing - Werte bleiben kontinuierlich im [0,1] Intervall
