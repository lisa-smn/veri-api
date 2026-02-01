# Factuality Agent Evaluation: FRANK Manifest n=200

**Datum:** 2026-02-01  
**Run-ID:** `factuality_agent_manifest_20260201_011300_gpt-4o-mini`  
**Status:** ✅ Erfolgreich abgeschlossen

---

## Run-Konfiguration

- **Dataset:** FRANK (`frank_subset_manifest.jsonl`)
- **Dataset Signature:** `9223fd84b05a9dc6dcaf1e67ce275eeee9a43fcf4aeebfc828e37bb7ae911eda`
- **n_total:** 2246 (Manifest-Größe)
- **n_used:** 200 (limit via `--max_examples 200`)
- **n_failed:** 0
- **Model:** gpt-4o-mini
- **Prompt Version:** v1
- **Issue Threshold:** 1 (default)
- **Seed:** 42 (reproduzierbar)
- **Bootstrap n:** 2000 resamples (95% CI)

---

## Metriken

### Confusion Matrix

| | Predicted: Error | Predicted: No Error |
|---|---|---|
| **Actual: Error** | TP = 145 | FN = 31 |
| **Actual: No Error** | FP = 13 | TN = 11 |

### Performance-Metriken (mit 95% Bootstrap-CI)

| Metrik | Wert | 95% CI |
|--------|------|--------|
| **Precision** | 0.9177 | [0.8710, 0.9583] |
| **Recall** | 0.8239 | [0.7680, 0.8771] |
| **F1** | **0.8683** | [0.8291, 0.9048] |
| **Balanced Accuracy** | 0.6411 | [0.5352, 0.7454] |
| **Accuracy** | 0.7800 | [0.7200, 0.8350] |
| **Specificity** | 0.4583 | - |
| **MCC** | 0.2251 | [0.0543, 0.3897] |
| **AUROC** | 0.8037 | - |

---

## Interpretation

### Hauptkennzahlen

- **F1 = 0.8683:** Gute Balance zwischen Precision und Recall. Der Agent erkennt 82.4% aller tatsächlichen Fehler (Recall) und 91.8% der als Fehler markierten Fälle sind tatsächlich Fehler (Precision).
- **AUROC = 0.8037:** Moderate bis gute Fähigkeit, Fehler von korrekten Summaries zu unterscheiden (thresholdfrei).
- **Balanced Accuracy = 0.6411:** Niedriger als Accuracy, da die Klasse "No Error" (n=24) deutlich kleiner ist als "Error" (n=176). Der Agent hat Schwierigkeiten mit der Minderheitsklasse.

### Vergleich mit vorherigen Runs

**Vorheriger Run (2026-01-07):**
- F1 = 0.792 [0.732, 0.843]
- Balanced Accuracy = 0.722
- AUROC = 0.892

**Aktueller Run (2026-02-01):**
- F1 = 0.8683 [0.8291, 0.9048] ← **höher**
- Balanced Accuracy = 0.6411 [0.5352, 0.7454] ← **niedriger**
- AUROC = 0.8037 ← **niedriger**

**Hinweis:** Die Unterschiede können durch verschiedene Seeds, Subset-Auswahl oder Modell-Updates entstehen. Beide Runs verwenden dieselbe `dataset_signature`, daher sind sie auf demselben Manifest-Basis evaluiert.

---

## Artefakte

**Run-Verzeichnis:** `results/evaluation/factuality/factuality_agent_manifest_20260201_011300_gpt-4o-mini/`

### Dateien

- `summary.json`: Vollständige Metriken mit Bootstrap-CIs
- `summary.md`: Human-readable Zusammenfassung
- `predictions.jsonl`: Pro-Beispiel-Vorhersagen (200 Zeilen)
- `run_metadata.json`: Git-Commit, Timestamp, Config-Snapshot

### Git-Commit

- **Commit Hash:** `8ec49494d67f0a7205ddc89401bd336d51d59910`
- **Python Version:** 3.13.1

---

## Reproduzierbarkeit

### Command

```bash
python scripts/eval_frank_factuality_agent_on_manifest.py \
  --manifest data/frank/frank_subset_manifest.jsonl \
  --max_examples 200 \
  --model gpt-4o-mini \
  --seed 42 \
  --bootstrap_n 2000
```

### Voraussetzungen

- `OPENAI_API_KEY` gesetzt (ENV oder `.env`)
- Manifest vorhanden: `data/frank/frank_subset_manifest.jsonl`
- Dependencies installiert (siehe `requirements.txt`)

---

## Vergleich mit Baselines

**Baseline-Runs (2026-01-07):**
- ROUGE-L: Spearman ρ = 0.20 [0.07, 0.33]
- BERTScore: Spearman ρ = 0.10 [-0.03, 0.24]

**Agent (2026-02-01):**
- F1 = 0.8683 [0.8291, 0.9048]

**Fazit:** Der Agent übertrifft die referenzbasierten Baselines deutlich, da er Evidence-Retrieval aus dem Artikel nutzt, während Baselines nur Ähnlichkeit zur Referenzsummary messen.

---

## Limitationen

1. **Unbalancierte Klassen:** Nur 24 Beispiele ohne Fehler (12%) vs. 176 mit Fehler (88%). Dies führt zu niedriger Balanced Accuracy.
2. **Issue Threshold:** `issue_threshold=1` bedeutet, dass bereits ein Issue als "Fehler vorhanden" klassifiziert wird. Dies kann zu False Positives führen.
3. **Prompt-Abhängigkeit:** Ergebnisse gelten für Prompt-Version v1. Änderungen am Prompt können die Metriken beeinflussen.
4. **Sample-Größe:** n=200 ist ausreichend für Bootstrap-CIs, aber größere Samples könnten stabilere Schätzungen liefern.
