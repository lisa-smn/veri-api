# Evaluation Datensätze - FRANK & FineSumFact

## Übersicht

Für die Factuality-Evaluation stehen zwei Datensätze zur Verfügung:

### 1. FRANK
- **Format:** Binär (has_error: true/false)
- **Test-Set:** 2246 Beispiele
- **Verwendung:** Haupt-Evaluation
- **Config:** `evaluation_configs/factuality_frank_test_v1.json`
- **Stichprobe für BA:** 300 Beispiele

### 2. FineSumFact
- **Format:** Binär (has_error: true/false) - **identisch zu FRANK**
- **Test-Set:** 693 Beispiele
- **Verwendung:** Ergänzende Evaluation
- **Config:** `evaluation_configs/factuality_finesumfact_test_v1.json`
- **Stichprobe für BA:** 200 Beispiele

## Beide Datensätze evaluieren

### Option 1: Separate Runs (Empfohlen)

**FRANK:**
```bash
python3 scripts/eval_factuality_structured.py \
  evaluation_configs/factuality_frank_test_v1.json \
  --dataset-path data/frank/frank_clean.jsonl
```

**FineSumFact:**
```bash
python3 scripts/eval_factuality_structured.py \
  evaluation_configs/factuality_finesumfact_test_v1.json \
  --dataset-path data/finesumfact/human_label_test_clean.jsonl
```

### Option 2: Beide nacheinander

```bash
# FRANK
./START_EVALUATION_BA.sh

# FineSumFact
python3 scripts/eval_factuality_structured.py \
  evaluation_configs/factuality_finesumfact_test_v1.json \
  --dataset-path data/finesumfact/human_label_test_clean.jsonl
```

## Warum beide Datensätze?

### FRANK
- ✅ Größeres Test-Set (mehr statistische Power)
- ✅ Gut etabliert in der Literatur
- ✅ Haupt-Fokus für BA

### FineSumFact
- ✅ Ergänzende Perspektive
- ✅ Andere Datenquelle (MediaSum)
- ✅ Zusätzliche Validierung

## Ergebnisse kombinieren

Nach beiden Runs hast du:

1. **FRANK-Ergebnisse:**
   - `results/evaluation/runs/docs/factuality_frank_test_v1.md`
   - Metriken für 300 Beispiele

2. **FineSumFact-Ergebnisse:**
   - `results/evaluation/runs/docs/factuality_finesumfact_test_v1.md`
   - Metriken für 200 Beispiele

3. **Vergleich:**
   ```bash
   python3 scripts/compare_runs.py \
     --run1 factuality_frank_test_v1 \
     --run2 factuality_finesumfact_test_v1
   ```

## Für BA-Text

**Empfehlung:**
- **Haupt-Evaluation:** FRANK (300 Beispiele)
- **Ergänzend:** FineSumFact (200 Beispiele) - zeigt Robustheit über Datensätze

**In BA-Text:**
- Haupt-Ergebnisse basieren auf FRANK
- FineSumFact als zusätzliche Validierung erwähnen
- Vergleich der Metriken zwischen beiden Datensätzen

## Cache-Status

Beide Datensätze haben separate Caches:
- FRANK: `results/evaluation/factuality/cache_gpt-4o-mini_v3_uncertain_spans.jsonl`
- FineSumFact: Wird automatisch erstellt beim ersten Run

## Zeitaufwand

- **FRANK (300 Beispiele):** ~15-30 Minuten (mit Cache: ~5-10 Min)
- **FineSumFact (200 Beispiele):** ~10-20 Minuten (mit Cache: ~3-7 Min)
- **Gesamt:** ~25-50 Minuten (mit Cache: ~8-17 Min)






