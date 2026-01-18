# Evidence-Gate Balanced Approach

## Problem-Analyse

**Erster Test (nach Schritt 1 & 2):**
- Recall: 1.000 ✅ (perfekt!)
- Specificity: 0.000 ❌ (schlecht, aber besser als 0.000 Recall)
- Precision: 0.940 ✅
- F1: 0.969 ✅

**Spätere Tests (nach weiteren Verbesserungen):**
- Recall: 0.000 ❌ (katastrophal!)
- Specificity: 1.000 ✅ (perfekt, aber nutzlos wenn Recall 0 ist)
- Precision: 0.000 ❌
- F1: 0.000 ❌

**Fazit:** Der erste Test war deutlich besser! Die späteren Änderungen haben das Problem verschlimmert.

## Ursache der Verschlechterung

1. **Test-Script Änderung:** "Uncertain" Issues zählen nicht mehr als "has_error"
   - ✅ Das ist gut (reduziert False Positives)
   - ❌ Aber wenn ALLE Claims zu "uncertain" werden, wird nichts erkannt

2. **Evidence-Gate zu strikt:** Alle "incorrect" werden zu "uncertain"
   - ❌ Das führt zu Recall 0.000

## Neue Balanced Approach

**Strategie:** Lockern für besseren Recall, aber trotzdem FP-Reduktion

- **Wenn LLM "incorrect" sagt:**
  - **Mit Evidence:** Behalte "incorrect" mit voller Confidence ✅
  - **Ohne Evidence, Confidence >= 0.5:** Behalte "incorrect", aber reduziere Confidence (um 30%, mindestens 0.5)
  - **Ohne Evidence, Confidence < 0.5:** Downgrade zu "uncertain" ❌

**Code:**
```python
if label == "incorrect":
    if not evidence_found:
        if conf < 0.5:
            # Sehr niedrige Confidence ohne Evidence => downgrade zu uncertain
            label = "uncertain"
            conf = min(conf, 0.4)
        else:
            # Confidence >= 0.5: Behalte "incorrect", aber reduziere Confidence
            conf = max(0.5, conf * 0.7)  # Reduziere um 30%, aber mindestens 0.5
    else:
        # Evidence gefunden: Behalte incorrect mit voller Confidence
        pass
```

## Erwartete Effekte

- **Recall sollte hoch bleiben** (>= 0.85, wie im ersten Test)
- **Specificity sollte steigen** (weil "uncertain" Issues nicht als "has_error" zählen)
- **Niedrige Confidence "incorrect" ohne Evidence** werden zu "uncertain" (FP-Reduktion)
- **Hohe Confidence "incorrect" ohne Evidence** bleiben "incorrect" (Recall-Erhaltung)

## Nächste Schritte

1. **Test nochmal laufen lassen:**
   ```bash
   python3 scripts/test_evidence_gate_eval.py
   ```

2. **Erwartete Ergebnisse:**
   - Recall: >= 0.85 (wie im ersten Test)
   - Specificity: > 0.000 (weil "uncertain" Issues nicht als "has_error" zählen)
   - Balanced Accuracy: > 0.500






