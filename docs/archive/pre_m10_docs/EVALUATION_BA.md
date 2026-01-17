# Evaluation für Bachelorarbeit

## Konfiguration

**Run ID:** `factuality_frank_test_v1`

**Stichprobe:** 300 Beispiele (repräsentativ für BA)

**Cache:** Aktiviert (beschleunigt Evaluation)

**Dataset:** FRANK Test-Set

## Starten

```bash
# Option 1: Mit Script
./START_EVALUATION_BA.sh

# Option 2: Direkt
python3 scripts/eval_factuality_structured.py \
  evaluation_configs/factuality_frank_test_v1.json \
  --dataset-path data/frank/frank_clean.jsonl
```

## Warum 300 Beispiele?

- **Repräsentativ:** Ausreichend für statistisch belastbare Metriken
- **Zeitlich machbar:** ~15-30 Minuten (mit Cache deutlich schneller)
- **Kosten:** Angemessen für BA (ca. 300 LLM-Calls)
- **Qualität:** Genug für aussagekräftige Ergebnisse

## Cache-Nutzung

Falls bereits ein Cache vorhanden ist:
- Bereits evaluierte Beispiele werden übersprungen
- Nur neue Beispiele werden neu evaluiert
- Deutlich schneller und kostengünstiger

**Cache-Location:** `results/evaluation/factuality/cache_gpt-4o-mini_v3_uncertain_spans.jsonl`

## Ergebnisse

Nach Abschluss findest du:

1. **Vollständige Dokumentation**
   ```
   results/evaluation/runs/docs/factuality_frank_test_v1.md
   ```
   - Run-Definition
   - Execution-Log
   - Metriken
   - Analyse
   - Interpretation

2. **Metriken (JSON)**
   ```
   results/evaluation/runs/results/factuality_frank_test_v1.json
   ```
   - Accuracy, Precision, Recall, F1
   - AUROC
   - Confusion Matrix

3. **Example-Level Results**
   ```
   results/evaluation/runs/results/factuality_frank_test_v1_examples.jsonl
   ```
   - Pro Beispiel: GT, Prediction, Score, Issue Spans

4. **Analyse**
   ```
   results/evaluation/runs/analyses/factuality_frank_test_v1.json
   ```
   - Robustheit (Threshold-Sweep)
   - Fehleranalyse (FP/FN Patterns)
   - Subset-Analyse

## Dauer

- **Ohne Cache:** ~15-30 Minuten (300 Beispiele)
- **Mit Cache:** ~5-10 Minuten (nur neue Beispiele)

## Für BA-Text verwenden

Die automatisch generierte Dokumentation enthält:
- ✅ Metriken (tabellarisch)
- ✅ Interpretation (automatisch generiert)
- ✅ Fehleranalyse (FP/FN Patterns)

**Ergänzen:**
- Fallstudien (manuell aus Example-Level Results)
- Diskussion der Ergebnisse
- Vergleich mit Baselines (falls vorhanden)

## Nächste Schritte

1. **Evaluation starten:** `./START_EVALUATION_BA.sh`
2. **Ergebnisse prüfen:** Dokumentation öffnen
3. **Fallstudien auswählen:** Aus Example-Level Results
4. **BA-Text schreiben:** Basierend auf Dokumentation

