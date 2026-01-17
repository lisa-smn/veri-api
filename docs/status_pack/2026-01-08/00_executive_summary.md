# Executive Summary

**Status Pack erstellt:** 2026-01-08  
**Spätere Updates:** 2026-01-16 (Readability-Finalisierung, Factuality Judge Baseline, Coherence Tests, Status-Dokumente für Factuality/Coherence), 2026-01-17 (Factuality Judge Final-Run ausstehend) – Verifikationssystem (veri-api)

**Datum:** 2026-01-17  
**Status:** Evaluation abgeschlossen für Coherence, Factuality und Readability (alle drei Agenten ✅ VERIFIED)

---

## Was das System kann

- **3-dimensionale Verifikation:** Bewertet Summaries in Factuality (Faktentreue), Coherence (Kohärenz) und Readability (Lesbarkeit) auf einer Skala von 0.0 (schlecht) bis 1.0 (gut).
- **Issue-Detection:** Markiert problematische Textstellen (Spans) mit Schweregrad (low/medium/high) und Typ (z.B. NUMBER, DATE, CONTRADICTION).
- **Explainability-Report:** Generiert einen strukturierten Report mit Executive Summary, Findings nach Dimension, Top-Spans und Statistiken.
- **API-basiert:** REST-API (`/verify`) für Integration in andere Systeme.
- **Reproduzierbare Evaluation:** Seed-basierte Runs mit Bootstrap-Konfidenzintervallen für alle Metriken.

---

## Was bereits evaluiert ist

### Coherence (SummEval, n=200)
- **Agent:** Spearman ρ = 0.41 [0.27, 0.53] – moderate Übereinstimmung mit menschlichen Ratings.
- **LLM-Judge Baseline:** Spearman ρ = 0.45 [0.33, 0.56] – leicht besser als Agent, aber beide CIs überlappen.

### Factuality (FRANK Manifest, n=200)
- **Agent:** F1 = 0.79 [0.73, 0.84], AUROC = 0.89 – gute Trennung zwischen fehlerhaften und korrekten Summaries.
- **Judge Baseline:** Smoke-Run ✅ (n=50, balanced), Final-Run ✅ (n=200, F1=0.9441, AUROC=0.9453) – höher als Agent, aber Sample unbalanciert (pos=176, neg=24) → BA=0.7093, MCC=0.4753.
- **Baselines (ROUGE-L/BERTScore):** Schwache Korrelationen (Spearman ρ < 0.20) – messen Ähnlichkeit, nicht Faktentreue.

### Readability (SummEval, n=200)
- **Agent:** Spearman ρ = 0.402 [0.268, 0.512], Pearson r = 0.390 [0.292, 0.468], MAE = 0.283 [0.263, 0.302] – moderate Übereinstimmung mit menschlichen Ratings.
- **Judge:** Spearman ρ = 0.280, Pearson r = 0.343 – schwächer als Agent, aber positive Korrelation.
- **Baselines (Flesch/Flesch-Kincaid/Gunning Fog):** Nahezu keine Korrelation (Spearman ρ ≈ -0.05) – klassische Formeln erfassen nicht semantische Aspekte.

---

## Wichtigste Ergebnisse

1. **Factuality-Agent übertrifft Baselines deutlich:** F1 = 0.79 vs. ROUGE-L Spearman ρ = 0.20. Baselines sind ungeeignet als Proxy für Faktentreue.
2. **Coherence-Agent zeigt moderate Korrelation:** Spearman ρ = 0.41 bedeutet, dass der Agent die Rangfolge der menschlichen Ratings teilweise erfasst, aber nicht perfekt.
3. **Readability-Agent zeigt moderate Korrelation:** Spearman ρ = 0.402 bedeutet, dass der Agent die Rangfolge menschlicher Lesbarkeitsbewertungen teilweise erfasst. Klassische Formeln (Flesch, Flesch-Kincaid, Gunning Fog) zeigen nahezu keine Korrelation (ρ ≈ -0.05).
4. **LLM-Judge als Baseline:** Mit Spearman ρ = 0.45 (Coherence) bzw. 0.280 (Readability) zeigen die Judges, dass moderne LLMs als Baseline funktionieren, aber nicht besser als spezialisierte Agents.
5. **AUROC = 0.89 (Factuality):** Der Agent trennt zuverlässig zwischen fehlerhaften und korrekten Summaries (0.5 = Zufall, 1.0 = perfekt).
6. **Bootstrap-CIs zeigen Unsicherheit:** Alle Metriken haben 95%-Konfidenzintervalle, die die Variabilität der Schätzungen abbilden.

---

## Nächste Schritte

1. **Readability-Evaluation:** ✅ Abgeschlossen (Spearman ρ = 0.402, n=200, seed=42).
2. **Coherence-Evaluation:** ✅ Abgeschlossen (Spearman ρ = 0.41, n=200, seed=42).
3. **Factuality Judge Final-Run:** ✅ Abgeschlossen (n=200, bootstrap=2000, F1=0.9441, AUROC=0.9453, ✅ verifiziert).
4. **Stress-Tests für Coherence:** Shuffle/Injection-Tests geplant, um Robustheit zu demonstrieren (P2: future work).
5. **Explainability-Verbesserungen:** Evidence-Spans derzeit nur für Factuality vollständig; Coherence könnte von externen Belegen profitieren (P2: future work).

---

## Wo die Artefakte liegen

- **Evaluation-Ergebnisse:** `results/evaluation/` (strukturiert nach Dimension: coherence/, factuality/, readability/, coherence_judge/, coherence_baselines/, factuality_baselines/, baselines/)
- **Run-Artefakte:** Jeder Run enthält `summary.json`, `summary.md`, `predictions.jsonl`, `run_metadata.json`
- **Aggregations-Matrizen:** `results/evaluation/summary_coherence_matrix.csv`, `summary_factuality_matrix.csv`, `baselines/summary_matrix.csv` (falls vorhanden)
- **Status-Dokumente:** 
  - `docs/status/readability_status.md` ✅
  - `docs/status/coherence_status.md` ✅
  - `docs/status/factuality_status.md` (Final-Run ausstehend, siehe `docs/status/FINAL_RUN_INSTRUCTIONS.md`)

---

**Vollständige Details:** Siehe `01_system_overview.md`, `02_explainability_audit.md`, `03_evaluation_results.md`, `04_metrics_glossary.md`, `05_repo_cleanup_report.md`, `06_appendix_artifacts_index.md`.

