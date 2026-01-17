# Eigene Einschätzung zur Evaluation

**Datum:** 2026-01-08  
**Autor:** Projekt-Entwickler  
**Zweck:** Kritische Bewertung der Evaluationsergebnisse für Betreuerin und Thesis

---

## Kontext & Ziel der Evaluation

Die Evaluation umfasst zwei Dimensionen: **Coherence** (auf SummEval, n=200) und **Factuality** (auf FRANK Manifest, n=200). Gemessen wird die Übereinstimmung mit menschlichen Ratings bzw. Goldlabels sowie der Vergleich zu klassischen Baselines (ROUGE-L, BERTScore) und einem modernen LLM-as-a-Judge-Ansatz. Ziel ist es, die Leistungsfähigkeit der Agenten zu quantifizieren und ihre Stärken sowie Grenzen zu identifizieren.

---

## Positive Punkte

**Reproduzierbarkeit:** Alle Runs sind seed-basiert (z.B. `seed=42`), enthalten vollständige Metadaten (`run_metadata.json` mit Git-Commit, Python-Version, Config) und Bootstrap-Konfidenzintervalle (n=2000) für alle Metriken. Dies ermöglicht eine exakte Reproduktion der Ergebnisse und eine transparente Unsicherheitsschätzung.

**Fairness:** Für Factuality wurde ein Manifest-basiertes System implementiert (`frank_subset_manifest.jsonl`), das Agent und Baselines auf exakt derselben Subset evaluiert. Die `dataset_signature` (SHA256-Hash) verifiziert programmatisch, dass beide Methoden identische Daten verwenden. Dies eliminiert einen häufigen Bias in Evaluationsvergleichen.

**Explainability-Mehrwert:** Die Agenten liefern nicht nur Scores, sondern strukturierte `issue_spans` mit `severity` (low/medium/high), `issue_type` (z.B. NUMBER, DATE, CONTRADICTION) und optional Evidence-Quotes. Das Explainability-Modul aggregiert diese zu einem nachvollziehbaren Report (`ExplainabilityResult`), der für Debugging und Nutzer-Feedback wertvoll ist.

**Robustheit/QA:** Dependency-Guards verhindern, dass Baseline-Runs mit fehlenden Paketen stillschweigend 0.0-Werte liefern (`dependencies_ok` in `run_metadata.json`). Aggregatoren (`aggregate_coherence_runs.py`, `aggregate_factuality_runs.py`) warnen bei inkonsistenten `dataset_signature`s und markieren ungültige Runs. Dies erhöht die Zuverlässigkeit der Evaluationsergebnisse.

---

## Ergebnisinterpretation

**Coherence-Agent** (Run: `coherence_20260107_205123_gpt-4o-mini_v1_seed42`): Spearman ρ = 0.41 [0.27, 0.53] bedeutet, dass der Agent die Rangfolge der menschlichen Ratings (1-5) teilweise erfasst. Pearson r = 0.35 [0.17, 0.53] zeigt eine moderate lineare Korrelation. MAE = 0.18 [0.16, 0.20] bedeutet, dass der Agent im Durchschnitt um 0.18 Punkte (auf Skala 0-1) von den menschlichen Ratings abweicht. R² = 0.04 zeigt, dass nur 4% der Varianz erklärt werden – das Modell ist deutlich besser als Zufall, aber weit von perfekt entfernt.

**Coherence-Judge** (Run: `coherence_judge_20260107_234710_gpt-4o-mini_v1_n3_seed42`): Spearman ρ = 0.45 [0.33, 0.56] ist leicht höher als der Agent, aber die CIs überlappen. R² = -0.12 ist negativ, was bedeutet, dass der Judge schlechter als eine Mittelwert-Baseline ist – ein Kalibrierungsproblem. Der Judge nutzt eine Rubrik (temp=0, n_judgments=3) und dient als moderne Baseline, nicht als Ground Truth.

**Factuality-Agent** (Run: `factuality_agent_manifest_20260107_215431_gpt-4o-mini`): F1 = 0.79 [0.73, 0.84] zeigt eine gute Balance zwischen Precision (0.79) und Recall (0.80). AUROC = 0.89 bedeutet, dass der Agent zuverlässig zwischen fehlerhaften und korrekten Summaries trennt (0.5 = Zufall, 1.0 = perfekt). Balanced Accuracy = 0.72 [0.66, 0.79] berücksichtigt die leichte Imbalance (TP=99, FP=27, TN=49, FN=25). MCC = 0.45 [0.31, 0.57] zeigt eine moderate Korrelation, die robuster als F1 bei Imbalance ist.

**Factuality-Baselines** (Runs: `factuality_rouge_l_20260107_230519_seed42`, `factuality_bertscore_20260107_230523_seed42`): ROUGE-L erreicht Spearman ρ = 0.20 [0.07, 0.33], BERTScore nur ρ = 0.10 [-0.03, 0.24] (CI schließt 0 ein). Beide haben R² < 0, was bedeutet, dass sie schlechter als eine Mittelwert-Baseline sind. **Kurzvergleich:** Baselines messen Ähnlichkeit zur Referenzsummary, nicht Faktentreue zum Artikel. Der Agent nutzt Evidence-Retrieval aus dem Artikel und ist daher für Faktentreue geeigneter. Spearman misst Rangfolge (wichtig für Ranking-Aufgaben), während F1/AUROC absolute Klassifikations-Performance messen (wichtig für binäre Entscheidungen).

---

## Limitations / Risiken

**SummEval Rater Noise & Subset-Größe:** Die Evaluation nutzt n=200 von 1700 verfügbaren Beispielen. Menschliche Ratings (1-5) haben intrinsische Varianz (Rater-Noise), was die Unsicherheit erhöht. Die Bootstrap-CIs (z.B. Spearman ρ = 0.41 [0.27, 0.53]) quantifizieren diese Unsicherheit, aber größere Samples würden engere CIs liefern.

**R² Interpretation:** Negatives R² (z.B. Judge: -0.12, BERTScore: -1.04) bedeutet, dass das Modell schlechter als "immer Mittelwert vorhersagen" ist. Dies ist ein Kalibrierungsproblem: Die Scores sind möglicherweise auf der falschen Skala oder haben einen systematischen Bias. R² sollte nicht isoliert betrachtet werden – Spearman ρ ist robuster gegen Kalibrierungsprobleme.

**FRANK binäre Labels:** FRANK liefert nur binäre Labels (`has_error=True/False`), keinen Fehlergrad. Dies erschwert die Evaluation feiner Scores (0.0-1.0), da der Agent kontinuierliche Scores liefert, aber nur binäre Ground Truth vorhanden ist. Der Threshold (`issue_threshold=1`) ist daher kritisch: Ein anderer Threshold würde andere F1/Precision/Recall-Werte liefern.

**Threshold-/Policy-Abhängigkeiten:** Der Factuality-Agent nutzt `issue_threshold=1` (ein Issue = fehlerhaft). Die Severity-Mapping (low/medium/high → Score) und die "uncertain"-Policy (wie werden unsichere Claims behandelt?) sind weitere Policy-Entscheidungen, die die Ergebnisse beeinflussen. Diese sollten in zukünftigen Arbeiten systematisch variiert werden.

**LLM-as-judge:** Der Judge hat intrinsische Bias/Varianz (selbst bei temp=0 und n_judgments=3). Er dient als moderne Baseline für Vergleichbarkeit, aber nicht als Ground Truth. Die Rubrik-Version (`v1`) ist eine weitere Policy-Entscheidung, die die Ergebnisse beeinflusst.

---

## Konkrete nächste Schritte

**Coherence:** Die Baselines (ROUGE-L, BERTScore) konnten aufgrund fehlender Referenzen nicht ausgewertet werden. Stress-Tests (Shuffle, Injection) sind geplant, um Robustheit zu demonstrieren. Die Aggregationstabelle (`summary_coherence_matrix.csv`) sollte finalisiert werden, um Agent, Judge und Baselines direkt vergleichen zu können.

**Readability:** Die Evaluationsstrategie muss festgelegt werden: Gibt es Goldlabels (z.B. in SummEval oder einem anderen Datensatz)? Falls nicht, sollte ein Proxy-Metrik (z.B. Flesch-Kincaid) oder ein LLM-Judge als Baseline genutzt werden. Der Agent ist implementiert und liefert Scores, aber ohne Evaluation ist die Qualität unklar.

**Optional als Future Work:** Kalibrierung der Scores (nur mit Train/Val-Split möglich, aktuell kein Split vorhanden). Judge nur bei "uncertain"-Claims nutzen (aktuell werden alle Claims vom Agent verarbeitet, Judge könnte als Fallback dienen).

---

## Abschluss

Der aktuelle Evaluationsstand ist vorzeigbar, weil er **nachvollziehbare, reproduzierbare Evaluation** mit Bootstrap-CIs, Manifest-basierter Fairness und vollständigen Artefakten liefert. Die **erklärbare Fehlerdiagnose** (Explainability-Modul mit `issue_spans`, `severity`, `issue_types`) ermöglicht es, nicht nur zu messen, sondern auch zu verstehen, warum der Agent bestimmte Scores liefert. Die Ergebnisse zeigen klare Stärken (Factuality F1=0.79, AUROC=0.89) und ehrliche Limitations (Coherence R²=0.04, Judge R²=-0.12), was für eine wissenschaftlich saubere Evaluation essentiell ist.

---

**Referenzen:**
- Coherence Agent: `results/evaluation/coherence/coherence_20260107_205123_gpt-4o-mini_v1_seed42/summary.json`
- Coherence Judge: `results/evaluation/coherence_judge/coherence_judge_20260107_234710_gpt-4o-mini_v1_n3_seed42/summary.json`
- Factuality Agent: `results/evaluation/factuality/factuality_agent_manifest_20260107_215431_gpt-4o-mini/summary.json`
- Factuality Baselines: `results/evaluation/factuality_baselines/factuality_rouge_l_20260107_230519_seed42/summary.json`, `factuality_bertscore_20260107_230523_seed42/summary.json`

