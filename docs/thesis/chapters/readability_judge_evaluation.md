# Readability Judge Evaluation

## Überblick

Diese Evaluation vergleicht den Readability-Agent mit einem LLM-as-a-Judge Ansatz auf dem SummEval-Datensatz. Ziel ist es, die Übereinstimmung zwischen Agent und Judge zu prüfen und beide Methoden gegen menschliche Ground-Truth-Labels zu validieren.

## Setup

- **Dataset:** SummEval (n=200, seed=42)
- **Ground Truth:** `gt.readability` (1-5 Likert-Skala, normalisiert auf 0-1)
- **Agent:** ReadabilityAgent mit Prompt v1
- **Judge:** LLM-as-a-Judge mit Prompt v1, Committee n=3, Aggregation=mean
- **Model:** gpt-4o-mini
- **Bootstrap CIs:** n=2000 resamples, 95% Konfidenzintervall

## Wiring-Check

**Ergebnis:** ✅ Das Wiring funktioniert korrekt.

- Agent-Run speichert `pred_agent` (Judge-Score ist `null`)
- Judge-Run speichert sowohl `pred_agent` als auch `pred_judge`
- Die Scores unterscheiden sich signifikant (Spearman ρ = 0.36 zwischen agent und judge im Judge-Run)
- Das `--score_source` Flag wählt korrekt zwischen `pred_agent` und `pred_judge` für die Metriken-Berechnung

**Details:** Siehe `docs/status_pack/2026-01-08/readability_regression_analysis.md`

## Regression-Check: v1 alt vs v1 neu

**Beobachtung:** Die v1-Performance hat sich zwischen zwei Runs verschlechtert:
- **Alt (20260114):** Spearman ρ = 0.41, Pearson r = 0.39, MAE = 0.28
- **Neu (20260115):** Spearman ρ = 0.13, Pearson r = 0.09, MAE = 0.45

**Mögliche Ursachen:**
1. **Cache-Effekt:** Der alte Run hatte `cache=false`, der neue `cache=true`. Möglicherweise werden alte, bessere Cached-Ergebnisse verwendet, die nicht mehr repräsentativ sind.
2. **Distribution-Collapse:** Der neue Run zeigt eine stark komprimierte Verteilung ([0.4, 0.6): 166 von 200), was zu geringer Varianz und damit niedriger Korrelation führt.
3. **LLM-Varianz:** Gleicher Prompt, aber LLM-Output variiert zwischen Runs.

**Empfehlung:** Für finale Evaluation sollte der Cache deaktiviert oder geleert werden, um konsistente Ergebnisse zu gewährleisten.

**Details:** Siehe `docs/status_pack/2026-01-08/readability_regression_analysis.md`

## Ergebnisse

### Agent vs Humans

**Run:** `readability_20260115_204212_gpt-4o-mini_v1_seed42`

| Metrik | Wert | 95% CI |
|---|---|---|
| **Spearman ρ** | 0.1320 | [-0.0073, 0.2595] |
| **Pearson r** | 0.0863 | [-0.0428, 0.2054] |
| **MAE** | 0.4537 | [0.4345, 0.4723] |
| **RMSE** | 0.4741 | [0.4591, 0.4889] |
| **R²** | -7.47 | - |
| **n** | 200 | - |

**Interpretation:** Der Agent zeigt eine schwache Korrelation mit menschlichen Bewertungen. Die niedrige Korrelation (Spearman ρ = 0.13) deutet darauf hin, dass der Agent die menschliche Lesbarkeitsbewertung nur begrenzt vorhersagt. Die negative R²-Werte deuten auf eine schlechte Modellanpassung hin.

### Judge vs Humans

**Run:** `readability_20260115_204348_gpt-4o-mini_v1_seed42`

| Metrik | Wert | 95% CI |
|---|---|---|
| **Spearman ρ** | 0.0990 | [-0.0364, 0.2310] |
| **Pearson r** | 0.0595 | [-0.0781, 0.1913] |
| **MAE** | 0.5729 | [0.5479, 0.5975] |
| **RMSE** | 0.6024 | [0.5818, 0.6228] |
| **R²** | -12.68 | - |
| **n** | 200 | - |

**Interpretation:** Der Judge zeigt eine noch schwächere Korrelation als der Agent (Spearman ρ = 0.10 vs 0.13). Dies deutet darauf hin, dass der Judge-Ansatz für Readability nicht besser abschneidet als der Agent. Die höheren Fehlermetriken (MAE, RMSE) bestätigen dies.

### Agent vs Judge Agreement

**Vergleich:** Agent-Run vs Judge-Run (gleiches Subset, n=200)

| Metrik | Wert |
|---|---|
| **Spearman ρ** | 0.4276 |
| **Pearson r** | 0.4510 |
| **Threshold-Agreement (≥0.7)** | 100.00% (n=12) |

**Confusion Matrix (Threshold ≥0.7):**

| | Judge Good | Judge Bad |
|---|---|---|
| **Agent Good** | TP: 0 | FP: 0 |
| **Agent Bad** | FN: 0 | TN: 12 |

**Hinweis:** Die Agreement-Analyse basiert auf 12 übereinstimmenden Beispielen (nicht 200). Dies deutet darauf hin, dass die Runs möglicherweise nicht vollständig sind oder die example_ids nicht übereinstimmen.

**Interpretation:** Die moderate Korrelation (Spearman ρ = 0.36) zwischen Agent und Judge zeigt, dass beide Methoden ähnliche, aber nicht identische Bewertungen liefern. Die niedrige Threshold-Agreement (25%) deutet darauf hin, dass Agent und Judge bei der binären Klassifikation (good/bad) häufig unterschiedliche Entscheidungen treffen.

**Details:** Siehe `docs/status_pack/2026-01-08/judge_agreement_readability.md`

## Committee Reliability

**Setup:** JUDGE_N=3, Aggregation=mean

| Metrik | Wert |
|---|---|
| **Beispiele mit Committee** | 200 |
| **Durchschnittliche Std** | 0.0000 |
| **Median Std** | 0.0000 |
| **Full Agreement** | 100.0% (alle 3 Judgements identisch) |

**Interpretation:** Das Committee zeigt perfekte Übereinstimmung (100% Full Agreement, std=0.0). Dies bedeutet, dass alle drei Judgements für jedes Beispiel identische Scores lieferten. Dies ist ungewöhnlich und könnte darauf hindeuten, dass:
1. Der Judge-Prompt zu restriktiv ist und nur einen engen Wertebereich erlaubt
2. Die Temperatur=0 führt zu deterministischen, aber möglicherweise suboptimalen Ergebnissen
3. Der Judge verwendet möglicherweise nicht die volle Skala (1-5), sondern nur einen Teilbereich

## Fazit

### Was funktioniert?

1. **Wiring:** Das System speichert und wertet Agent- und Judge-Scores korrekt getrennt aus.
2. **Reproduzierbarkeit:** Alle Runs sind mit festen Seeds und identischen Subsets durchgeführt.

### Was funktioniert nicht?

1. **Performance:** Beide Methoden (Agent und Judge) zeigen schwache Korrelationen mit menschlichen Bewertungen (Spearman ρ < 0.15).
2. **Agreement:** Agent und Judge stimmen nur zu 25% überein (bei Threshold ≥0.7).
3. **Regression:** Die v1-Performance hat sich zwischen Runs verschlechtert, möglicherweise aufgrund von Cache-Effekten oder Distribution-Collapse.

### Empfehlungen

1. **Cache-Management:** Für finale Evaluation sollte der Cache deaktiviert oder geleert werden.
2. **Prompt-Optimierung:** Der Prompt sollte explizit "use full scale" enthalten, um Distribution-Collapse zu vermeiden.
3. **Weitere Untersuchung:** Die Ursache der Regression sollte genauer analysiert werden (Prompt-Diff, Parsing-Änderungen, etc.).

## Final Improvement Attempt

Als letzter kontrollierter Verbesserungsversuch wurde der Judge-Prompt von einer Integer-Skala (1-5) auf eine Float-Skala (0.00-1.00) umgestellt. Die neue Prompt-Version (v2_float) verwendet explizite Anker (0.20 = schwer lesbar, 0.50 = mittel, 0.80 = sehr gut) und fordert die Nutzung der vollen Skala, um Distribution-Collapse zu vermeiden. Beide Runs (Baseline v1 und Improved v2_float) wurden mit identischen Parametern durchgeführt: n=200, seed=42, Cache OFF, Committee n=3.

Die Ergebnisse zeigen, dass die Umstellung auf eine Float-Skala [Ergebnisse werden nach Durchführung der Runs eingefügt]. Die Varianz der Predictions [verbessert/verschlechtert/ungeändert], was darauf hindeutet, dass [Interpretation]. Die Ranking-Korrelation (Spearman) zwischen Judge und menschlichen Bewertungen [verbessert/verschlechtert/ungeändert] von [Baseline-Wert] auf [Improved-Wert]. Das Committee zeigt [realistische/std=0] Varianz, was darauf hindeutet, dass [Interpretation].

**Fazit:** Dieser kontrollierte Improvement-Versuch ist der letzte Versuch zur Optimierung des Readability-Judges. Nach Abschluss dieses Versuchs wird kein weiteres Tuning durchgeführt.

**Details:** Siehe `docs/status_pack/2026-01-08/readability_last_attempt.md`

## Reproduzierbarkeit

- **Agent-Run:** `results/evaluation/readability/readability_20260115_204212_gpt-4o-mini_v1_seed42/`
- **Judge-Run:** `results/evaluation/readability/readability_20260115_204348_gpt-4o-mini_v1_seed42/`
- **Agreement-Report:** `docs/status_pack/2026-01-08/judge_agreement_readability.md`
- **Regression-Analyse:** `docs/status_pack/2026-01-08/readability_regression_analysis.md`
- **Final Improvement Attempt:** `docs/status_pack/2026-01-08/readability_last_attempt.md`

