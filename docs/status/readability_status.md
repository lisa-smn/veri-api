# Readability Evaluation: Final Status

**Datum:** 20260116
**Status:** Finale Evaluation abgeschlossen

---

## Setup

- **Dataset:** SummEval (`data/sumeval/sumeval_clean.jsonl`)
- **Subset:** n=200, seed=42 (reproduzierbar)
- **Model:** gpt-4o-mini
- **Prompt:** v1
- **Bootstrap:** n=2000 resamples, 95% Konfidenzintervall
- **Cache:** off (cache_hits=0, cache_misses=200)
- **Score Source:** agent (Agent-Score als primäre Metrik)

---

## System vs Humans: Vergleichstabelle

| System | Spearman ρ (95% CI) | Pearson r (95% CI) | MAE (95% CI) | RMSE (95% CI) | R² | n |
|---|---|---|---|---|---|---|
| **Agent** | 0.402 [0.268, 0.512] | 0.390 [0.292, 0.468] | 0.283 [0.263, 0.302] | 0.316 [0.300, 0.332] | -2.773 | 200 |
| **Judge** | 0.280 | 0.343 | 0.417 | 0.446 | -6.492 | 200 |
| **Flesch** | -0.054 [-0.197, 0.085] | 0.168 [-0.090, 0.386] | 0.384 [0.362, 0.405] | 0.414 [0.393, 0.433] | -4.061 | 200 |
| **Flesch-Kincaid** | -0.055 [-0.199, 0.093] | 0.124 [-0.115, 0.337] | 0.448 [0.425, 0.473] | 0.483 [0.461, 0.505] | -5.908 | 200 |
| **Gunning Fog** | -0.039 [-0.172, 0.101] | 0.047 [-0.125, 0.213] | 0.579 [0.548, 0.609] | 0.622 [0.594, 0.649] | -10.443 | 200 |

**Hinweis:** Judge-Metriken wurden aus `predictions.jsonl` berechnet (keine Bootstrap-CIs verfügbar). ROUGE/BERTScore/BLEU/METEOR nicht berechenbar, da SummEval keine Referenz-Zusammenfassungen enthält.

---

## Interpretation

### Hauptkennzahl: Spearman ρ (Rangkorrelation)

**Warum Spearman primär ist:**
- Spearman ρ misst die **Rangfolge** (monotone Beziehung), nicht absolute Werte
- Robust gegen Skalenfehler und nicht-lineare Beziehungen
- Für Ranking-basierte Evaluation ideal: "Ist A besser als B?" ist wichtiger als "Ist A genau 0.8?"
- Agent zeigt moderate Rangkorrelation (ρ = 0.402), was bedeutet, dass der Agent die Rangfolge menschlicher Bewertungen teilweise erfasst

### Agent vs Judge vs Baselines

- **Agent (ρ = 0.402):** Beste Performance. Moderate Korrelation mit menschlichen Bewertungen, zeigt dass der Agent semantische und strukturelle Aspekte der Lesbarkeit erfasst.
- **Judge (ρ = 0.280):** Schwächer als Agent, aber immer noch positive Korrelation. LLM-as-a-Judge zeigt, dass moderne LLMs als Baseline funktionieren, aber nicht besser als der spezialisierte Agent.
- **Klassische Formeln (ρ ≈ -0.05):** Nahezu keine Korrelation. Flesch, Flesch-Kincaid und Gunning Fog basieren nur auf statistischen Textmerkmalen (Satzlänge, Silbenanzahl) und erfassen nicht die semantischen Aspekte, die menschliche Bewerter berücksichtigen.

### R² negativ: Was bedeutet das?

**R² = -2.773 (Agent) bedeutet:**
- Das Modell ist **schlechter als eine Mittelwert-Baseline** (immer den Durchschnitt vorhersagen)
- **Nicht widersprüchlich zu brauchbarem Spearman:** R² misst absolute Werte, Spearman misst Rangfolge
- **Mögliche Ursachen:**
  - Kalibrierungsproblem: Agent-Scores sind nicht auf der gleichen Skala wie GT
  - Geringe GT-Varianz: Wenn alle GT-Werte ähnlich sind, ist R² instabil
  - Skalenfehler: Agent könnte systematisch zu hoch/niedrig vorhersagen, aber die Rangfolge stimmt

**Fazit:** Für Ranking-basierte Evaluation (Spearman) ist der Agent brauchbar, auch wenn R² negativ ist.

### MAE in verständlicher Skala

**MAE auf Skala 0-1:** 0.283
**MAE auf Skala 1-5:** 1.13 Punkte

Der Agent weicht im Durchschnitt um etwa 1.13 Punkte (auf der 1-5 Skala) von den menschlichen Bewertungen ab.

---

## Reproduzierbarkeit

### Git-Tag (Reproducible Version)

Diese Evaluation ist reproduzierbar ab Git-Tag `readability-final-2026-01-16`:

```bash
# Annotated Tag erstellen (falls noch nicht vorhanden):
git tag -a readability-final-2026-01-16 -m "Readability package final (docs/tests/ci)"

# Tag pushen:
git push origin readability-final-2026-01-16

# Reproduktion: Checkout des Tags
git checkout readability-final-2026-01-16
```

**Hinweis:** Der Tag sollte auf dem Commit `558e17442542d9a1d5034895c7afb1b35f2d675b` (oder später) erstellt werden, der die finale Evaluation enthält.

### Run-Artefakte

- **Agent-Run:** `results/evaluation/readability/readability_20260116_170832_gpt-4o-mini_v1_seed42/`
  - Git-Commit: `558e17442542d9a1d5034895c7afb1b35f2d675b`
  - Timestamp: 20260116_170832
  - Seed: 42
  - n_used: 200/1700

- **Baseline-Runs:**
  - `results/evaluation/baselines/baselines_readability_flesch_fk_fog_20260116_175246_seed42/`
  - `results/evaluation/baselines/baselines_readability_coherence_flesch_fk_fog_20260116_175253_seed42/`

- **Aggregation:**
  - `results/evaluation/baselines/summary_matrix.csv`
  - `results/evaluation/baselines/summary_matrix.md`

### Vollständige Reproduktion

#### Schritt 1: Umgebungsvariablen setzen

```bash
export ENABLE_LLM_JUDGE=true
export JUDGE_MODE=secondary
export JUDGE_N=3
export JUDGE_TEMPERATURE=0
export OPENAI_API_KEY="your-key-here"  # Falls LLM-Calls benötigt
```

#### Schritt 2: Agent-Run (mit Judge)

```bash
python scripts/eval_sumeval_readability.py \
  --data data/sumeval/sumeval_clean.jsonl \
  --max_examples 200 \
  --seed 42 \
  --prompt_version v1 \
  --bootstrap_n 2000 \
  --cache_mode off \
  --score_source agent
```

**Output:** `results/evaluation/readability/readability_YYYYMMDD_HHMMSS_gpt-4o-mini_v1_seed42/`

#### Schritt 3: Baselines evaluieren

```bash
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
# Option 1: Metriken in Anführungszeichen setzen (empfohlen)
--metrics 'flesch,fk,fog'

# Option 2: noglob verwenden
noglob python scripts/eval_sumeval_baselines.py --metrics flesch,fk,fog ...
```

**Output:** `results/evaluation/baselines/baselines_readability_flesch_fk_fog_YYYYMMDD_HHMMSS_seed42/`

#### Schritt 4: Baselines aggregieren

```bash
python scripts/aggregate_baseline_runs.py \
  --baseline_dir results/evaluation/baselines/ \
  --out_csv results/evaluation/baselines/summary_matrix.csv \
  --out_md results/evaluation/baselines/summary_matrix.md
```

**Output:** `results/evaluation/baselines/summary_matrix.csv` und `summary_matrix.md`

#### Schritt 5: Status-Report generieren

```bash
# Ersetze <AGENT_RUN_DIR> mit dem tatsächlichen Run-Verzeichnis
python scripts/build_readability_status.py \
  --agent_run_dir results/evaluation/readability/<AGENT_RUN_DIR> \
  --baseline_matrix results/evaluation/baselines/summary_matrix.csv \
  --out docs/status/readability_status.md \
  --plot  # Optional: Erstellt Scatter-Plot (wenn matplotlib verfügbar)
```

**Output:** `docs/status/readability_status.md` und optional `docs/status/img/readability_scatter.png`

#### Schritt 6: Sanity-Checks (optional)

```bash
# Prüft Artefakt-Links und Script-Ausführbarkeit
python scripts/check_readability_package.py

# Prüft, ob Status-Report konsistent ist
python scripts/build_readability_status.py \
  --agent_run_dir results/evaluation/readability/<AGENT_RUN_DIR> \
  --baseline_matrix results/evaluation/baselines/summary_matrix.csv \
  --out docs/status/readability_status.md \
  --check
```

---

## Artefakte

### Run-Verzeichnisse

- **Agent:** `results/evaluation/readability/readability_20260116_170832_gpt-4o-mini_v1_seed42/`
- **Baselines:** `results/evaluation/baselines/baselines_readability_flesch_fk_fog_20260116_175246_seed42/`

### Aggregations-Matrizen

- **CSV:** `results/evaluation/baselines/summary_matrix.csv`
- **Markdown:** `results/evaluation/baselines/summary_matrix.md`

### Vergleichs-Report

- **Thesis-Kapitel:** `docs/thesis/chapters/classical_metrics_baselines.md`

---

## Tests & Qualitätssicherung

Das Readability-Paket verfügt über automatisierte Tests:

- **Contract-Tests:** Prüfen Score-Range [0, 1] und Robustheit gegen Edge-Cases (leere Strings, Unicode, sehr lange Texte)
- **Mapping-Tests:** Validieren Normalisierung 1-5 ↔ 0-1
- **Judge Secondary Tests:** Prüfen Agent-primär/Judge-Fallback-Logik
- **Determinismus-Tests:** Stellen sicher, dass gleicher Input identischen Output liefert
- **Integration-Tests:** Mini-Test für `eval_sumeval_readability.py` (ohne LLM-Calls)

**Ausführung:**
```bash
# Alle Readability-Tests
pytest tests/readability/ -v

# Sanity-Checks (Artefakt-Links, Script-Ausführbarkeit)
python scripts/check_readability_package.py

# Vollständige Checks (Tests + Sanity + Status-Report)
./scripts/check_readability_all.sh
```

**Hinweis:** LLM-Judge wird in Tests gemockt (keine echten API Calls).

**Skipped Tests:**
- `test_eval_sumeval_readability_output_structure`: Wird in CI übersprungen, da Subprocess-Ausführung in CI-Sandbox nicht zuverlässig ist. Der Test prüft nur die CLI-Struktur via `--help` und wird lokal ausgeführt, wenn Subprocess verfügbar ist.

---

**Details zu Metriken:** Siehe `docs/status_pack/2026-01-08/04_metrics_glossary.md`
**Vollständige Evaluation:** Siehe `docs/status_pack/2026-01-08/03_evaluation_results.md`