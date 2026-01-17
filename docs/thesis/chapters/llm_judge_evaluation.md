# LLM-as-a-Judge Evaluation

## Was ist LLM-as-a-Judge?

LLM-as-a-Judge ist eine Bewertungsstrategie, bei der ein Large Language Model (LLM) als "Gutachter" eingesetzt wird, um die Qualität von Textzusammenfassungen zu bewerten. Im Gegensatz zu Ground-Truth-Labels (menschliche Bewertungen) ist der Judge ein automatisiertes System, das basierend auf einer definierten Rubrik und striktem JSON-Output bewertet.

**Wichtig:** Der Judge ist **nicht** Ground Truth, sondern dient als:
- **Comparator:** Vergleichsbaseline für den Agent
- **Robustness-Check:** Prüfung der Stabilität der Bewertung
- **Sekundäre Bewertung:** Ergänzung zum primären Agent-Score

## Setup

### Datasets
- **SummEval:** Readability und Coherence (kontinuierliche Skala 1-5, normalisiert auf 0-1)
  - n=200, seed=42
- **FRANK:** Factuality (binäre Klassifikation: has_error / no_error)
  - n=200, seed=42 (identisches Subset via Manifest)

### Judge-Konfiguration
- **Modell:** gpt-4o-mini
- **Prompt-Version:** v1
- **Committee:** n=3 (mehrere Judgements für Robustheit)
- **Aggregation:** mean (Mittelwert der normalisierten Scores)
- **Temperatur:** 0 (für Konsistenz)
- **Mode:** secondary (Agent-Score bleibt primär, Judge-Score wird zusätzlich gespeichert)

### Metriken
- **Kontinuierlich (Readability/Coherence):** Pearson r, Spearman ρ, MAE, RMSE, R² (mit Bootstrap-CIs, n=2000)
- **Binär (Factuality):** Precision, Recall, F1, Accuracy, Balanced Accuracy, Specificity, MCC, AUROC (mit Bootstrap-CIs)

## Ergebnisse SummEval

### Readability

#### Agent vs Humans
- **Spearman ρ:** [Wert aus Run] (95% CI: [lower, upper])
- **Pearson r:** [Wert aus Run] (95% CI: [lower, upper])
- **MAE:** [Wert aus Run] (95% CI: [lower, upper])
- **RMSE:** [Wert aus Run] (95% CI: [lower, upper])
- **R²:** [Wert aus Run]

#### Judge vs Humans
- **Spearman ρ:** [Wert aus Run] (95% CI: [lower, upper])
- **Pearson r:** [Wert aus Run] (95% CI: [lower, upper])
- **MAE:** [Wert aus Run] (95% CI: [lower, upper])
- **RMSE:** [Wert aus Run] (95% CI: [lower, upper])
- **R²:** [Wert aus Run]

#### Agent vs Judge Agreement
- **Spearman ρ:** [Wert aus Agreement-Report]
- **Pearson r:** [Wert aus Agreement-Report]
- **Threshold-Agreement (≥0.7):** [%] Agreement
- **Confusion Matrix:** [TP, FP, FN, TN]

### Coherence

#### Agent vs Humans
- **Spearman ρ:** [Wert aus Run] (95% CI: [lower, upper])
- **Pearson r:** [Wert aus Run] (95% CI: [lower, upper])
- **MAE:** [Wert aus Run] (95% CI: [lower, upper])
- **RMSE:** [Wert aus Run] (95% CI: [lower, upper])
- **R²:** [Wert aus Run]

#### Judge vs Humans
- **Spearman ρ:** [Wert aus Run] (95% CI: [lower, upper])
- **Pearson r:** [Wert aus Run] (95% CI: [lower, upper])
- **MAE:** [Wert aus Run] (95% CI: [lower, upper])
- **RMSE:** [Wert aus Run] (95% CI: [lower, upper])
- **R²:** [Wert aus Run]

#### Agent vs Judge Agreement
- **Spearman ρ:** [Wert aus Agreement-Report]
- **Pearson r:** [Wert aus Agreement-Report]
- **Threshold-Agreement (≥0.7):** [%] Agreement
- **Confusion Matrix:** [TP, FP, FN, TN]

## Ergebnisse FRANK

### Factuality

#### Agent vs Goldlabels
- **F1:** [Wert aus Run] (95% CI: [lower, upper])
- **Precision:** [Wert aus Run] (95% CI: [lower, upper])
- **Recall:** [Wert aus Run] (95% CI: [lower, upper])
- **Balanced Accuracy:** [Wert aus Run] (95% CI: [lower, upper])
- **MCC:** [Wert aus Run] (95% CI: [lower, upper])
- **AUROC:** [Wert aus Run]

#### Judge vs Goldlabels
- **F1:** [Wert aus Run] (95% CI: [lower, upper])
- **Precision:** [Wert aus Run] (95% CI: [lower, upper])
- **Recall:** [Wert aus Run] (95% CI: [lower, upper])
- **Balanced Accuracy:** [Wert aus Run] (95% CI: [lower, upper])
- **MCC:** [Wert aus Run] (95% CI: [lower, upper])
- **AUROC:** [Wert aus Run]

#### Agent vs Judge Agreement
- **Threshold-Agreement (has_error):** [%] Agreement
- **Confusion Matrix:** [TP, FP, FN, TN]

## Reliability

### Committee-Stabilität (JUDGE_N=3)

#### Readability
- **Durchschnittliche Std:** [Wert]
- **Median Std:** [Wert]
- **Full Agreement:** [%] (alle 3 Judgements identisch)

#### Coherence
- **Durchschnittliche Std:** [Wert]
- **Median Std:** [Wert]
- **Full Agreement:** [%] (alle 3 Judgements identisch)

**Interpretation:** Eine niedrige Standardabweichung deutet auf stabile, konsistente Bewertungen hin. Full Agreement zeigt, dass alle drei Judgements identisch waren.

## Fazit

### Wann ist Judge nützlich?

1. **Comparator:** Der Judge dient als moderne Baseline zum Vergleich mit dem Agent. Wenn Agent und Judge ähnliche Ergebnisse liefern, deutet dies auf Robustheit hin.

2. **Robustness-Check:** Committee-basierte Bewertungen (n>1) zeigen, wie stabil die Bewertung ist. Niedrige Varianz deutet auf konsistente Bewertungen hin.

3. **Sekundäre Bewertung:** Im "secondary"-Mode wird der Judge-Score zusätzlich zum Agent-Score gespeichert, ohne den primären Score zu überschreiben. Dies ermöglicht eine spätere Analyse der Übereinstimmung.

### Grenzen

1. **Nicht Ground Truth:** Der Judge ist kein Ersatz für menschliche Bewertungen. Er dient als automatisiertes Vergleichsinstrument.

2. **LLM-Bias:** Der Judge kann denselben Bias wie der Agent haben (z.B. systematische Überschätzung/Unterschätzung).

3. **Kosten:** Mehrere Judgements (Committee) erhöhen die API-Kosten.

4. **Parsing-Risiko:** Trotz striktem JSON-Output kann Parsing fehlschlagen (mit Retry-Mechanismus abgefangen).

## Reproduzierbarkeit

Alle Runs sind mit festen Seeds (42) und identischen Subsets durchgeführt. Die Run-IDs und Pfade sind in den jeweiligen `summary.json` und `run_metadata.json` dokumentiert.

### Run-IDs (Beispiele)
- **Readability Agent:** `readability_YYYYMMDD_HHMMSS_gpt-4o-mini_v1_seed42`
- **Readability Judge:** `readability_YYYYMMDD_HHMMSS_gpt-4o-mini_v1_seed42` (mit `ENABLE_LLM_JUDGE=true`, `--score_source judge`)
- **Coherence Agent:** `coherence_YYYYMMDD_HHMMSS_gpt-4o-mini_v1_seed42`
- **Coherence Judge:** `coherence_YYYYMMDD_HHMMSS_gpt-4o-mini_v1_seed42` (mit `ENABLE_LLM_JUDGE=true`, `--score_source judge`)

### Agreement-Reports
- **Readability:** `docs/status_pack/2026-01-08/judge_agreement_readability.md`
- **Coherence:** `docs/status_pack/2026-01-08/judge_agreement_coherence.md`

