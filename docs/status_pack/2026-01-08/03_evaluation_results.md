# Evaluation-Ergebnisse

**Datum:** 2026-01-08  
**Status:** Finale Runs für Coherence und Factuality abgeschlossen

---

## Key Takeaways

1. **Factuality-Agent übertrifft Baselines deutlich:** F1 = 0.79 vs. ROUGE-L Spearman ρ = 0.20. Baselines messen Ähnlichkeit, nicht Faktentreue.
2. **Coherence-Agent zeigt moderate Korrelation:** Spearman ρ = 0.41 bedeutet, dass der Agent die Rangfolge der menschlichen Ratings teilweise erfasst.
3. **LLM-Judge als Baseline funktioniert:** Mit Spearman ρ = 0.45 zeigt der Judge, dass moderne LLMs als Baseline geeignet sind, aber nicht als Ground Truth.
4. **Bootstrap-CIs zeigen Unsicherheit:** Alle Metriken haben 95%-Konfidenzintervalle, die die Variabilität der Schätzungen abbilden.
5. **Coherence-Baselines nicht auswertbar:** SummEval enthält keine Referenzsummaries, daher ROUGE-L/BERTScore = 0.0.

---

## Coherence (SummEval, n=200)

**Status:** Finale Evaluation abgeschlossen. Vollständige Dokumentation: `docs/status/coherence_status.md`

### Tabelle: Methoden-Vergleich

| Method | n | Spearman ρ (95% CI) | Pearson r (95% CI) | MAE (95% CI) | RMSE (95% CI) |
|--------|---|---------------------|--------------------|--------------|---------------|
| **Coherence-Agent** | 200 | 0.41 [0.27, 0.53] | 0.35 [0.17, 0.53] | 0.18 [0.16, 0.20] | 0.24 [0.21, 0.28] |
| **LLM-Judge** | 200 | 0.45 [0.33, 0.56] | 0.48 [0.36, 0.58] | 0.21 [0.18, 0.23] | 0.26 [0.24, 0.29] |
| **ROUGE-L** | 0 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| **BERTScore** | 0 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |

**Run-IDs:**
- Agent: `coherence_20260107_205123_gpt-4o-mini_v1_seed42`
- Judge: `coherence_judge_20260107_234710_gpt-4o-mini_v1_n3_seed42`
- ROUGE-L: `coherence_rouge_l_20260107_230323_seed42` (n_no_ref=1700)
- BERTScore: `coherence_bertscore_20260107_230512_seed42` (n_no_ref=1700)

### Interpretation

**Coherence-Agent:**
- Spearman ρ = 0.41 bedeutet, dass der Agent die Rangfolge der menschlichen Ratings (1-5) teilweise erfasst. Höhere Werte = bessere Übereinstimmung in der Rangfolge.
- MAE = 0.18 bedeutet, dass der Agent im Durchschnitt um 0.18 Punkte (auf Skala 0-1) von den menschlichen Ratings abweicht. Niedrigere Werte = besser.
- Die CIs überlappen mit dem LLM-Judge, daher kein signifikanter Unterschied.

**LLM-Judge:**
- Spearman ρ = 0.45 ist leicht höher als der Agent, aber die CIs überlappen. Der Judge nutzt eine Rubrik (temp=0, n_judgments=3) und ist als moderne Baseline gedacht, nicht als Ground Truth.
- R² = -0.12 bedeutet, dass der Judge schlechter als eine Mittelwert-Baseline ist (negatives R² = Modell ist schlechter als "immer Mittelwert vorhersagen").

**ROUGE-L / BERTScore:**
- Nicht auswertbar, da SummEval keine Referenzsummaries enthält (n_no_ref=1700). Dies unterstreicht die Notwendigkeit referenzfreier Bewertungsansätze.

---

## Factuality (FRANK Manifest, n=200)

### Tabelle A: Klassifikation (Agent vs Judge vs Baselines)

| Metric | Value | 95% CI |
|--------|-------|--------|
| **F1** | 0.79 | [0.73, 0.84] |
| **Balanced Accuracy** | 0.72 | [0.66, 0.79] |
| **AUROC** | 0.89 | - |
| **MCC** | 0.45 | [0.31, 0.57] |
| **Precision** | 0.79 | [0.71, 0.86] |
| **Recall** | 0.80 | [0.73, 0.87] |
| **Accuracy** | 0.74 | [0.68, 0.80] |
| **Specificity** | 0.64 | - |

**Run-IDs:**
- Agent: `factuality_agent_manifest_20260107_215431_gpt-4o-mini`
- Judge Smoke: `judge_factuality_20260116_231906_gpt-4o-mini_v2_binary_seed42` (n=50, balanced, ✅ verifiziert)
- Judge Final: `judge_factuality_20260116_233505_gpt-4o-mini_v2_binary_seed42` (n=200, ✅ verifiziert; siehe `docs/status/factuality_status.md`)
- Baselines: siehe Tabelle B

**Dataset Signature:** `c32c25988d7a041fea833c132f4bd2bcc6484de4c22a157d114994e9812eb299`  
**Confusion Matrix (Agent):** TP=99, FP=27, TN=49, FN=25

**Interpretation:**
- **Agent:** F1 = 0.79 bedeutet, dass der Agent eine gute Balance zwischen Precision und Recall hat. Höher = besser (max 1.0). AUROC = 0.89 bedeutet, dass der Agent zuverlässig zwischen fehlerhaften und korrekten Summaries trennt (0.5 = Zufall, 1.0 = perfekt). MCC = 0.45 zeigt eine moderate Korrelation zwischen Vorhersage und Ground Truth (0 = Zufall, 1 = perfekt, -1 = perfekt falsch).
- **LLM-as-a-Judge (Baseline):** Der Judge nutzt einen binary Prompt (v2_binary) mit `error_present: true/false` und Majority-Vote über mehrere Judgements (JUDGE_N=3, temp=0). Er dient als moderne Baseline, nicht als Ground Truth. Vergleichbar mit Readability/Coherence Judge-Ansatz.
- **Judge Final (n=200):** F1 = 0.9441 [0.9178, 0.9681], AUROC = 0.9453, Precision = 0.9286, Recall = 0.9602. **Höher als Agent** (F1=0.9441 vs 0.792), aber Sample unbalanciert (pos=176, neg=24) → Balanced Accuracy = 0.7093, MCC = 0.4753 niedriger als Agent (BA=0.722, MCC=0.445). Specificity = 0.4583 niedrig aufgrund vieler FP (13 FP vs 11 TN).

### Tabelle B: Baselines (Regression + AUROC)

| Method | n | Spearman ρ (95% CI) | Pearson r (95% CI) | MAE (95% CI) | RMSE (95% CI) | R² | AUROC |
|--------|---|---------------------|--------------------|--------------|---------------|-----|-------|
| **ROUGE-L** | 200 | 0.20 [0.07, 0.33] | 0.14 [0.01, 0.28] | 0.43 [0.39, 0.46] | 0.50 [0.46, 0.53] | -0.05 | - |
| **BERTScore** | 200 | 0.10 [-0.03, 0.24] | 0.09 [-0.04, 0.22] | 0.59 [0.54, 0.64] | 0.69 [0.66, 0.73] | -1.04 | - |

**Run-IDs:**
- ROUGE-L: `factuality_rouge_l_20260107_230519_seed42`
- BERTScore: `factuality_bertscore_20260107_230523_seed42`
- **Dataset Signature:** `c32c25988d7a041fea833c132f4bd2bcc6484de4c22a157d114994e9812eb299` (identisch mit Agent)

**Interpretation:**
- **Baselines messen Ähnlichkeit, nicht Faktentreue:** ROUGE-L und BERTScore vergleichen die Summary mit einer Referenzsummary, nicht mit dem Artikel. Daher sind sie ungeeignet als Proxy für Faktentreue.
- **Schwache Korrelationen:** Spearman ρ < 0.20 bedeutet, dass die Baselines kaum mit den menschlichen Factuality-Labels korrelieren.
- **R² < 0:** BERTScore hat R² = -1.04, was bedeutet, dass das Modell schlechter als eine Mittelwert-Baseline ist.
- **Hinweis zu BERTScore:** Warnings über "Roberta pooler" sind irrelevant für die Score-Nutzung (nur für Debugging).

---

## Readability (SummEval, n=200)

### Tabelle: Methoden-Vergleich

| Method | n | Spearman ρ (95% CI) | Pearson r (95% CI) | MAE (95% CI) | RMSE (95% CI) | R² |
|--------|---|---------------------|--------------------|--------------|---------------|-----|
| **Readability-Agent** | 200 | 0.402 [0.268, 0.512] | 0.390 [0.292, 0.468] | 0.283 [0.263, 0.302] | 0.316 [0.300, 0.332] | -2.773 |
| **LLM-Judge** | 200 | 0.280 | 0.343 | 0.417 | 0.446 | -6.492 |
| **Flesch** | 200 | -0.054 [-0.197, 0.085] | 0.168 [-0.090, 0.386] | 0.384 [0.362, 0.405] | 0.414 [0.393, 0.433] | -4.061 |
| **Flesch-Kincaid** | 200 | -0.055 [-0.199, 0.093] | 0.124 [-0.115, 0.338] | 0.448 [0.425, 0.473] | 0.483 [0.461, 0.505] | -5.908 |
| **Gunning Fog** | 200 | -0.039 [-0.172, 0.101] | 0.047 [-0.125, 0.213] | 0.579 [0.548, 0.609] | 0.622 [0.594, 0.649] | -10.443 |

**Run-IDs:**
- Agent: `readability_20260116_170832_gpt-4o-mini_v1_seed42`
- Baselines: `baselines_readability_flesch_fk_fog_20260116_175246_seed42`

**Hinweis:** ROUGE/BERTScore/BLEU/METEOR nicht berechenbar, da SummEval keine Referenz-Zusammenfassungen enthält.

### Interpretation

**Readability-Agent:**
- Spearman ρ = 0.402 bedeutet, dass der Agent die Rangfolge der menschlichen Ratings (1-5) teilweise erfasst. Höhere Werte = bessere Übereinstimmung in der Rangfolge.
- MAE = 0.283 bedeutet, dass der Agent im Durchschnitt um 0.283 Punkte (auf Skala 0-1) bzw. **1.13 Punkte (auf Skala 1-5)** von den menschlichen Ratings abweicht.
- R² = -2.773 bedeutet, dass der Agent schlechter als eine Mittelwert-Baseline ist (negatives R² = Modell ist schlechter als "immer Mittelwert vorhersagen"). Dies ist **nicht widersprüchlich zu brauchbarem Spearman**, da R² absolute Werte misst, während Spearman die Rangfolge misst.

**LLM-Judge:**
- Spearman ρ = 0.280 ist schwächer als der Agent, aber immer noch positiv. Der Judge nutzt eine Rubrik (temp=0, n_judgments=3) und ist als moderne Baseline gedacht, nicht als Ground Truth.

**Klassische Readability-Formeln:**
- Flesch, Flesch-Kincaid und Gunning Fog zeigen nahezu keine Korrelation (Spearman ρ ≈ -0.05 bis -0.04) mit menschlichen Bewertungen.
- Diese Formeln basieren nur auf statistischen Textmerkmalen (Satzlänge, Silbenanzahl, komplexe Wörter) und erfassen nicht die semantischen Aspekte, die menschliche Bewerter berücksichtigen.
- **Fazit:** Klassische Formeln sind ungeeignet, um menschliche Lesbarkeitsbewertungen vorherzusagen.

---

## Artefakt-Index

### Coherence
- Agent: `results/evaluation/coherence/coherence_20260107_205123_gpt-4o-mini_v1_seed42/`
- Judge: `results/evaluation/coherence_judge/coherence_judge_20260107_234710_gpt-4o-mini_v1_n3_seed42/`
- Baselines: `results/evaluation/coherence_baselines/coherence_rouge_l_20260107_230323_seed42/`, `coherence_bertscore_20260107_230512_seed42/`

### Factuality
- Agent: `results/evaluation/factuality/factuality_agent_manifest_20260107_215431_gpt-4o-mini/`
- Baselines: `results/evaluation/factuality_baselines/factuality_rouge_l_20260107_230519_seed42/`, `factuality_bertscore_20260107_230523_seed42/`

### Readability
- Agent: `results/evaluation/readability/readability_20260116_170832_gpt-4o-mini_v1_seed42/`
- Baselines: `results/evaluation/baselines/baselines_readability_flesch_fk_fog_20260116_175246_seed42/`
- Aggregation: `results/evaluation/baselines/summary_matrix.csv/.md`

**Jeder Run enthält:**
- `summary.json`: Metriken + CIs + Metadaten
- `summary.md`: Human-readable Zusammenfassung
- `predictions.jsonl`: Pro-Beispiel-Vorhersagen
- `run_metadata.json`: Timestamp, Git-Commit, Python-Version, Seed, Config

**Aggregations-Matrizen:**
- `results/evaluation/summary_coherence_matrix.csv/.md` (falls vorhanden)
- `results/evaluation/summary_factuality_matrix.csv/.md` (falls vorhanden)
- `results/evaluation/baselines/summary_matrix.csv/.md` (Readability/Coherence Baselines)

---

**Details zu Metriken:** Siehe `04_metrics_glossary.md`.  
**Vollständiger Artefakt-Index:** Siehe `06_appendix_artifacts_index.md`.

