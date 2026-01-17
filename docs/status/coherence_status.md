# Coherence Evaluation: Final Status

**Datum:** 2026-01-16  
**Status:** Finale Evaluation abgeschlossen

---

## Setup

- **Dataset:** SummEval (`data/sumeval/sumeval_clean.jsonl`)
- **Subset:** n=200, seed=42 (reproduzierbar)
- **Model:** gpt-4o-mini
- **Prompt:** v1
- **Bootstrap:** n=2000 resamples, 95% Konfidenzintervall
- **Cache:** write (Agent), write/read (Judge, empfohlen)

---

## System vs Humans: Vergleichstabelle

| System | Spearman ρ (95% CI) | Pearson r (95% CI) | MAE (95% CI) | RMSE (95% CI) | R² | n |
|--------|---------------------|--------------------|--------------|---------------|-----|---|
| **Agent** | 0.409 [0.268, 0.534] | 0.345 [0.172, 0.529] | 0.178 [0.155, 0.202] | 0.241 [0.206, 0.277] | 0.042 | 200 |
| **Judge** | 0.450 [0.330, 0.560] | 0.480 [0.360, 0.580] | 0.210 [0.180, 0.230] | 0.260 [0.240, 0.290] | -0.120 | 200 |
| **ROUGE-L** | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 | 0 |
| **BERTScore** | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 | 0 |

**Hinweise:**
- **ROUGE-L / BERTScore:** Nicht auswertbar, da SummEval keine Referenzsummaries enthält (n_no_ref=1700). Dies unterstreicht die Notwendigkeit referenzfreier Bewertungsansätze.
- **SummaC/SummaCoz:** Referenzfreie Coherence-Baselines sind verfügbar, aber noch nicht integriert (P1: optional).

---

## Interpretation

### Hauptkennzahl: Spearman ρ (Rangkorrelation)

**Warum Spearman primär ist:**
- Spearman ρ misst die **Rangfolge** (monotone Beziehung), nicht absolute Werte
- Robust gegen Skalenfehler und nicht-lineare Beziehungen
- Für Ranking-basierte Evaluation ideal: "Ist A kohärenter als B?" ist wichtiger als "Ist A genau 0.8?"
- Agent zeigt moderate Rangkorrelation (ρ = 0.409), was bedeutet, dass der Agent die Rangfolge menschlicher Bewertungen teilweise erfasst

### Agent vs Judge vs Baselines

- **Agent (ρ = 0.409):** Moderate Korrelation mit menschlichen Bewertungen. Der Agent erfasst interne logische Konsistenz, Informationsfluss und Referenzen, aber nicht perfekt.
- **Judge (ρ = 0.450):** Leicht besser als Agent, aber die CIs überlappen (kein signifikanter Unterschied). Der Judge nutzt eine Rubrik (temp=0, n_judgments=3) und ist als moderne Baseline gedacht, nicht als Ground Truth.
- **ROUGE-L / BERTScore:** Nicht auswertbar (keine Referenzen). Selbst wenn Referenzen vorhanden wären, messen diese Metriken Ähnlichkeit, nicht Kohärenz.

### R² niedrig: Was bedeutet das?

**R² = 0.042 (Agent) bedeutet:**
- Das Modell erklärt nur 4.2% der Varianz in den menschlichen Bewertungen
- **Nicht widersprüchlich zu brauchbarem Spearman:** R² misst absolute Werte, Spearman misst Rangfolge
- **Mögliche Ursachen:**
  - Kalibrierungsproblem: Agent-Scores sind nicht auf der gleichen Skala wie GT
  - Geringe GT-Varianz: Wenn alle GT-Werte ähnlich sind, ist R² instabil
  - Skalenfehler: Agent könnte systematisch zu hoch/niedrig vorhersagen, aber die Rangfolge stimmt

**Fazit:** Für Ranking-basierte Evaluation (Spearman) ist der Agent brauchbar, auch wenn R² niedrig ist.

### MAE in verständlicher Skala

**MAE auf Skala 0-1:** 0.178  
**MAE auf Skala 1-5:** 0.71 Punkte

Der Agent weicht im Durchschnitt um etwa 0.71 Punkte (auf der 1-5 Skala) von den menschlichen Bewertungen ab.

---

## Reproduzierbarkeit

### Agent-Run

**Run-ID:** `coherence_20260107_205123_gpt-4o-mini_v1_seed42`  
**Pfad:** `results/evaluation/coherence/coherence_20260107_205123_gpt-4o-mini_v1_seed42/`

**Command:**
```bash
python scripts/eval_sumeval_coherence.py \
  --data data/sumeval/sumeval_clean.jsonl \
  --max_examples 200 \
  --seed 42 \
  --bootstrap_n 2000 \
  --cache_mode write
```

### Judge-Run

**Run-ID:** `coherence_judge_20260107_234710_gpt-4o-mini_v1_n3_seed42`  
**Pfad:** `results/evaluation/coherence_judge/coherence_judge_20260107_234710_gpt-4o-mini_v1_n3_seed42/`

**Command:**
```bash
ENABLE_LLM_JUDGE=true JUDGE_MODE=secondary JUDGE_N=3 JUDGE_TEMPERATURE=0 \
python scripts/eval_sumeval_coherence_llm_judge.py \
  --data data/sumeval/sumeval_clean.jsonl \
  --max_examples 200 \
  --seed 42 \
  --bootstrap_n 2000 \
  --cache_mode write
```

### Baseline-Runs

**ROUGE-L:**
- Run-ID: `coherence_rouge_l_20260107_230323_seed42`
- Pfad: `results/evaluation/coherence_baselines/coherence_rouge_l_20260107_230323_seed42/`
- Status: Nicht auswertbar (n_no_ref=1700)

**BERTScore:**
- Run-ID: `coherence_bertscore_20260107_230512_seed42`
- Pfad: `results/evaluation/coherence_baselines/coherence_bertscore_20260107_230512_seed42/`
- Status: Nicht auswertbar (n_no_ref=1700)

---

## Artefakte

### Agent-Run
- `results/evaluation/coherence/coherence_20260107_205123_gpt-4o-mini_v1_seed42/`
  - `summary.json`: Metriken + CIs
  - `summary.md`: Human-readable Zusammenfassung
  - `predictions.jsonl`: Pro-Beispiel-Vorhersagen
  - `run_metadata.json`: Timestamp, Git-Commit, Config

### Judge-Run
- `results/evaluation/coherence_judge/coherence_judge_20260107_234710_gpt-4o-mini_v1_n3_seed42/`
  - `summary.json`: Metriken + CIs
  - `summary.md`: Human-readable Zusammenfassung
  - `predictions.jsonl`: Pro-Beispiel-Vorhersagen
  - `run_metadata.json`: Timestamp, Git-Commit, Config

### Baselines
- `results/evaluation/coherence_baselines/` (ROUGE-L, BERTScore, nicht auswertbar)

---

## Thesis Framing

**Methodisch sinnvoll:**
- **Agent vs Judge:** Beide zeigen moderate Korrelation (ρ ≈ 0.41–0.45), was bedeutet, dass sowohl spezialisierte Agents als auch generische LLMs die Rangfolge menschlicher Bewertungen teilweise erfassen.
- **Referenzfreie Evaluation:** SummEval enthält keine Referenzen, daher sind ROUGE/BERTScore nicht auswertbar. Dies unterstreicht die Notwendigkeit referenzfreier Bewertungsansätze (Agent, Judge).
- **Ranking vs. absolute Werte:** Spearman ρ = 0.409 ist brauchbar für Ranking, auch wenn R² = 0.042 niedrig ist. Für praktische Anwendungen ist "Ist A kohärenter als B?" wichtiger als "Ist A genau 0.8?".

**Fazit:** Der Coherence-Agent zeigt moderate Übereinstimmung mit menschlichen Bewertungen und ist vergleichbar mit LLM-as-a-Judge. Referenzfreie Bewertungsansätze sind für Coherence essentiell.

---

## Offene Fragen / Future Work

- **SummaC/SummaCoz Baseline:** Referenzfreie Coherence-Baselines sind verfügbar, aber noch nicht integriert (P1: optional).
- **Stress-Tests:** Shuffle/Injection-Tests geplant, um Robustheit zu demonstrieren (P2: future work).
- **Prompt-Ablation:** v1 vs. alternative Prompts auf 50 Samples (P2: future work).

