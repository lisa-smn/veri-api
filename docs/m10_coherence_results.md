# M10 Coherence Evaluation Results

## Zusammenfassung

Evaluation des Coherence-Agenten auf SummEval-Dataset (Human Coherence Ratings).

**Settings:**
- Dataset: `data/sumeval/sumeval_clean.jsonl`
- Max Examples: 200
- Seed: 42
- Model: gpt-4o-mini
- Prompt: v1

**Run-ID (Agent):** `coherence_20260107_205123_gpt-4o-mini_v1_seed42`

---

## Vergleichstabelle: Agent vs. Baselines

| Methode | Pearson r (95% CI) | Spearman ρ (95% CI) | MAE (95% CI) | RMSE (95% CI) | n |
|---------|-------------------|---------------------|--------------|---------------|---|
| **Coherence-Agent** | 0.3455 [0.1720, 0.5287] | 0.4086 [0.2677, 0.5340] | 0.1777 [0.1551, 0.2017] | 0.2409 [0.2060, 0.2772] | 200 |
| ROUGE-L | N/A (keine Referenzen) | N/A (keine Referenzen) | N/A | N/A | 0 |
| BERTScore | N/A (keine Referenzen) | N/A (keine Referenzen) | N/A | N/A | 0 |

**Hinweis:** Die SummEval-Daten enthalten keine Referenzsummaries (`ref`/`reference`-Feld), daher konnten ROUGE-L und BERTScore nicht berechnet werden. Diese Baselines erfordern Referenzsummaries für den Vergleich (Summary vs. Reference).

---

## Stress-Tests

**Status:** Nicht ausgeführt (kann bei Bedarf nachträglich ausgeführt werden)

Die Stress-Tests (Shuffle, Injection) prüfen, ob der Agent "merkt", wenn Text kaputt gemacht wird:
- **Shuffle:** Permutiert Satzreihenfolge → erwartet: Score fällt
- **Injection:** Injiziert klar inkohärenten Satz → erwartet: Score fällt

**Ausführung:**
```bash
# Shuffle-Test
python3 scripts/stress_test_coherence.py \
  --data data/sumeval/sumeval_clean.jsonl \
  --max_examples 200 \
  --seed 42 \
  --model gpt-4o-mini \
  --mode shuffle

# Injection-Test
python3 scripts/stress_test_coherence.py \
  --data data/sumeval/sumeval_clean.jsonl \
  --max_examples 200 \
  --seed 42 \
  --model gpt-4o-mini \
  --mode inject
```

---

## Interpretation

Der Coherence-Agent zeigt moderate positive Korrelationen mit den Human Coherence Ratings:
- **Spearman ρ = 0.41** (95% CI: [0.27, 0.53]): Moderate monotone Beziehung
- **Pearson r = 0.35** (95% CI: [0.17, 0.53]): Moderate lineare Beziehung
- **MAE = 0.18** (95% CI: [0.16, 0.20]): Mittlere absolute Abweichung auf [0,1]-Skala
- **RMSE = 0.24** (95% CI: [0.21, 0.28]): Root Mean Squared Error

**Vergleich mit Baselines:** Da SummEval keine Referenzsummaries enthält, konnten klassische Baselines (ROUGE-L, BERTScore) nicht berechnet werden. Diese würden normalerweise die Ähnlichkeit zwischen generierter Summary und Referenz messen, was als Proxy für Coherence dient. Der Agent misst Coherence direkt durch Analyse von Textstruktur und logischer Konsistenz.

**Limitationen:**
- Keine Baseline-Vergleiche möglich (fehlende Referenzen)
- Stress-Tests noch nicht ausgeführt (können Robustheit des Agents demonstrieren)

---

## Artefakte

- **Agent-Run:** `results/evaluation/coherence/coherence_20260107_205123_gpt-4o-mini_v1_seed42/`
- **Baseline-Runs:** `results/evaluation/coherence_baselines/`
- **Aggregation:** `results/evaluation/coherence/summary_matrix.csv` / `.md`

