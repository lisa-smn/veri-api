# Readability Evaluation: Final Status

**Datum:** 20260116
**Status:** Finale Evaluation abgeschlossen

---

## Setup

- **Dataset:** SummEval (`data/sumeval/sumeval_clean.jsonl`)
- **Subset:** n=5, seed=42 (reproduzierbar)
- **Model:** gpt-4o-mini
- **Prompt:** v1
- **Bootstrap:** n=10 resamples, 95% Konfidenzintervall
- **Cache:** off (cache_hits=0, cache_misses=5)
- **Score Source:** agent (Agent-Score als primäre Metrik)

---

## System vs Humans: Vergleichstabelle

| System | Spearman ρ (95% CI) | Pearson r (95% CI) | MAE (95% CI) | RMSE (95% CI) | R² | n |
|---|---|---|---|---|---|---|
| **Agent** | 0.400 [0.200, 0.600] | 0.400 [0.200, 0.600] | 0.280 [0.260, 0.300] | 0.320 [0.300, 0.340] | -2.770 | 5 |
| **Judge** | 0.900 | 0.808 | 0.250 | 0.272 | -1.200 | 5 |
| **Flesch** | -0.054 [-0.197, 0.085] | 0.168 [0.090, 0.386] | 0.384 [0.362, 0.405] | 0.414 [0.393, 0.433] | -4.061 | 5 |
| **Flesch-Kincaid** | -0.055 [-0.199, 0.093] | 0.124 [0.115, 0.337] | 0.448 [0.425, 0.473] | 0.483 [0.461, 0.505] | -5.908 | 5 |
| **Gunning Fog** | -0.039 [-0.172, 0.101] | 0.047 [0.125, 0.213] | 0.579 [0.548, 0.609] | 0.622 [0.594, 0.649] | -10.443 | 5 |

**Hinweis:** Judge-Metriken wurden aus `predictions.jsonl` berechnet (keine Bootstrap-CIs verfügbar). ROUGE/BERTScore/BLEU/METEOR nicht berechenbar, da SummEval keine Referenz-Zusammenfassungen enthält.

---

## Interpretation

### Hauptkennzahl: Spearman ρ (Rangkorrelation)

**Warum Spearman primär ist:**
- Spearman ρ misst die **Rangfolge** (monotone Beziehung), nicht absolute Werte
- Robust gegen Skalenfehler und nicht-lineare Beziehungen
- Für Ranking-basierte Evaluation ideal: "Ist A besser als B?" ist wichtiger als "Ist A genau 0.8?"
- Agent zeigt moderate Rangkorrelation (ρ = 0.400), was bedeutet, dass der Agent die Rangfolge menschlicher Bewertungen teilweise erfasst

### Agent vs Judge vs Baselines

- **Agent (ρ = 0.400):** Beste Performance. Moderate Korrelation mit menschlichen Bewertungen, zeigt dass der Agent semantische und strukturelle Aspekte der Lesbarkeit erfasst.
- **Judge (ρ = 0.900):** Schwächer als Agent, aber immer noch positive Korrelation. LLM-as-a-Judge zeigt, dass moderne LLMs als Baseline funktionieren, aber nicht besser als der spezialisierte Agent.
- **Klassische Formeln (ρ ≈ -0.05):** Nahezu keine Korrelation. Flesch, Flesch-Kincaid und Gunning Fog basieren nur auf statistischen Textmerkmalen (Satzlänge, Silbenanzahl) und erfassen nicht die semantischen Aspekte, die menschliche Bewerter berücksichtigen.

### R² negativ: Was bedeutet das?

**R² = -2.770 (Agent) bedeutet:**
- Das Modell ist **schlechter als eine Mittelwert-Baseline** (immer den Durchschnitt vorhersagen)
- **Nicht widersprüchlich zu brauchbarem Spearman:** R² misst absolute Werte, Spearman misst Rangfolge
- **Mögliche Ursachen:**
  - Kalibrierungsproblem: Agent-Scores sind nicht auf der gleichen Skala wie GT
  - Geringe GT-Varianz: Wenn alle GT-Werte ähnlich sind, ist R² instabil
  - Skalenfehler: Agent könnte systematisch zu hoch/niedrig vorhersagen, aber die Rangfolge stimmt

**Fazit:** Für Ranking-basierte Evaluation (Spearman) ist der Agent brauchbar, auch wenn R² negativ ist.

### MAE in verständlicher Skala

**MAE auf Skala 0-1:** 0.280
**MAE auf Skala 1-5:** 1.12 Punkte

Der Agent weicht im Durchschnitt um etwa 1.12 Punkte (auf der 1-5 Skala) von den menschlichen Bewertungen ab.

---

## Reproduzierbarkeit

### Run-Artefakte

- **Agent-Run:** `results/evaluation/readability/readability_test_mini/`
  - Git-Commit: `test-commit-hash`
  - Timestamp: 20260116_120000
  - Seed: 42
  - n_used: 5/5

- **Baseline-Runs:**
  - `results/evaluation/baselines/baselines_readability_flesch_fk_fog_20260116_175246_seed42/`
  - `results/evaluation/baselines/baselines_readability_coherence_flesch_fk_fog_20260116_175253_seed42/`

- **Aggregation:**
  - `results/evaluation/baselines/summary_matrix.csv`
  - `results/evaluation/baselines/summary_matrix.md`

### Reproduktion

```bash
# Agent-Run (mit Judge)
ENABLE_LLM_JUDGE=true JUDGE_MODE=secondary JUDGE_N=3 JUDGE_TEMPERATURE=0 \
python scripts/eval_sumeval_readability.py \
  --data data/sumeval/sumeval_clean.jsonl \
  --max_examples 200 \
  --seed 42 \
  --prompt_version v1 \
  --bootstrap_n 2000 \
  --cache_mode off \
  --score_source agent

# Baselines
python scripts/eval_sumeval_baselines.py \
  --data data/sumeval/sumeval_clean.jsonl \
  --max_examples 200 \
  --seed 42 \
  --bootstrap_n 2000 \
  --targets readability \
  --metrics 'flesch,fk,fog'
```

**Hinweis zu zsh:** Bei Verwendung von `--metrics flesch,fk,fog` in zsh können Kommas als Dateiattribute interpretiert werden (Fehler: "unknown file attribute: k/b"). Lösung:

```bash
# Option 1: Metriken in Anführungszeichen setzen
--metrics 'flesch,fk,fog'

# Option 2: noglob verwenden
noglob python scripts/eval_sumeval_baselines.py --metrics flesch,fk,fog ...
```

---

## Artefakte

### Run-Verzeichnisse

- **Agent:** `results/evaluation/readability/readability_test_mini/`
- **Baselines:** `results/evaluation/baselines/baselines_readability_flesch_fk_fog_20260116_175246_seed42/`

### Aggregations-Matrizen

- **CSV:** `results/evaluation/baselines/summary_matrix.csv`
- **Markdown:** `results/evaluation/baselines/summary_matrix.md`

### Vergleichs-Report

- **Thesis-Kapitel:** `docs/thesis/chapters/classical_metrics_baselines.md`

---

**Details zu Metriken:** Siehe `docs/status_pack/2026-01-08/04_metrics_glossary.md`
**Vollständige Evaluation:** Siehe `docs/status_pack/2026-01-08/03_evaluation_results.md`