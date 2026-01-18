# Final Evaluation Stand

**Datum:** 2026-01-16  
**Git Tag:** `thesis-snapshot-2026-01-17`  
**Commit:** `558e17442542d9a1d5034895c7afb1b35f2d675b`  
**Status:** ✅ Alle drei Dimensionen evaluiert (Readability, Factuality, Coherence)

---

## Kern-Ergebnisse

| Dimension | Agent (Hauptkennzahl) | Judge | Klassische Baselines |
|-----------|----------------------|-------|---------------------|
| **Readability** | Spearman ρ = **0.402** [0.268, 0.512] | ρ = 0.280 | ρ ≈ -0.05 (Flesch/FK/Fog) |
| **Factuality** | F1 = **0.792** [0.732, 0.843], AUROC = 0.892 | F1 = 0.9441 | Spearman ρ < 0.20 (ROUGE/BERTScore) |
| **Coherence** | Spearman ρ = **0.409** [0.268, 0.534] | ρ = 0.450 | Nicht auswertbar (keine Referenzen) |

**n = 200** für alle Dimensionen (SummEval für Readability/Coherence, FRANK für Factuality)

---

## Was bedeutet das?

1. **Agenten übertreffen klassische Baselines deutlich:** Readability Agent (ρ = 0.402) vs. Flesch/FK/Fog (ρ ≈ -0.05). Klassische Formeln erfassen keine semantischen Aspekte.

2. **Agenten sind vergleichbar mit LLM-as-a-Judge:** Readability Agent besser (ρ = 0.402 vs 0.280), Coherence ähnlich (ρ = 0.409 vs 0.450, CIs überlappen), Factuality Agent niedriger F1 (0.792 vs 0.9441) aber höhere Balanced Accuracy (0.722 vs 0.7093).

3. **Spearman ρ ist primär:** Misst Rangfolge (robust gegen Skalenfehler), nicht absolute Werte. Für Ranking-basierte Evaluation ideal.

4. **Factuality AUROC = 0.892:** Der Agent trennt zuverlässig zwischen fehlerhaften und korrekten Summaries (0.5 = Zufall, 1.0 = perfekt).

5. **Bootstrap-CIs zeigen Unsicherheit:** Alle Metriken haben 95%-Konfidenzintervalle, die die Variabilität der Schätzungen abbilden.

---

## Datasets & Methoden

- **FRANK:** Factuality (Binary: `has_error` true/false), n=200, seed=42
- **SummEval:** Coherence + Readability (Kontinuierlich: 1-5, normalisiert zu 0-1), n=200, seed=42

**Methoden:**
- **Agenten:** Factuality, Coherence, Readability (strukturierte Analyse mit IssueSpans)
- **LLM-as-a-Judge:** GPT-4o-mini als Baseline (optional, `ENABLE_LLM_JUDGE=true`)
- **Klassische Baselines:** Flesch Reading Ease, Flesch-Kincaid, Gunning Fog (nur Readability)

---

## Reproduzierbarkeit

**Git Tags:**
- `readability-final-2026-01-16`
- `thesis-snapshot-2026-01-17`

**Run-Artefakte:**
- `results/evaluation/readability/readability_20260116_170832_gpt-4o-mini_v1_seed42/`
- `results/evaluation/factuality/judge_factuality_20260116_233505_gpt-4o-mini_v2_binary_seed42/`
- `results/evaluation/baselines/baselines_readability_flesch_fk_fog_20260116_175246_seed42/`

**Detail-Reports:**
- `docs/status/readability_status.md` - Readability: Agent vs Judge vs Baselines
- `docs/status/factuality_status.md` - Factuality: Agent vs Judge
- `docs/status/coherence_status.md` - Coherence: Agent vs Judge

**Vollständige Dokumentation:**
- `docs/milestones/M10_evaluation_setup.md` - Evaluationsansätze, Methoden, detaillierte Ergebnisse

---

## Limitationen

- **SummEval ohne Referenzen:** ROUGE/BERTScore nicht berechenbar für Readability/Coherence
- **Sample-Größe:** n=200 (ausreichend für Bootstrap-CIs)
- **Prompt-Abhängigkeit:** Ergebnisse gelten für v1/v3 Prompts

---

**Weitere Details:** Siehe `docs/milestones/M10_evaluation_setup.md` für vollständige Evaluationsansätze, Methoden und Interpretation.

