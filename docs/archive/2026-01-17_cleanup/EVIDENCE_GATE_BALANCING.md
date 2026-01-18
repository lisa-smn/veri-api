# Evidence-Gate Balancing - Recall vs. Specificity

## Problem
Nach den Verbesserungen:
- **Recall: 0.000** ❌ (keine echten Fehler erkannt)
- **Specificity: 1.000** ✅ (keine False Positives)
- **Alle Claims werden zu "uncertain"** (50/50)

**Ursache:** Evidence wird nicht gefunden, daher werden alle "incorrect" Claims zu "uncertain" downgraded.

## Implementierte Balancing-Strategie

### High-Confidence Hard Errors Exception ✅
Wenn das LLM "incorrect" mit:
- **Confidence >= 0.8** (sehr sicher)
- **Error-Type in (ENTITY, NUMBER, DATE)** (harte Fehler, nicht vage)
- **Mindestens etwas Evidence vorhanden** (auch wenn Coverage nicht perfekt)

→ Dann erlauben wir "incorrect" auch ohne perfekte Evidence.

**Code:**
```python
high_confidence_hard_error = (
    conf >= 0.8 and 
    error_type in ("ENTITY", "NUMBER", "DATE") and
    valid_evidence  # Mindestens etwas Evidence vorhanden
)

if not evidence_found and not high_confidence_hard_error:
    # Downgrade zu uncertain
else:
    # Behalte incorrect, aber reduziere Confidence leicht wenn Evidence nicht perfekt
    if not evidence_found:
        conf = min(conf, 0.75)  # Leicht reduzieren
```

## Erwartete Effekte

- **Recall sollte steigen** (von 0.000 auf > 0.70)
- **Specificity sollte hoch bleiben** (>= 0.20)
- **Nur harte, high-confidence Fehler** werden ohne perfekte Evidence erlaubt
- **Vage Fehler** bleiben weiterhin "uncertain"

## Nächste Schritte

1. **Test nochmal laufen lassen:**
   ```bash
   python3 scripts/test_evidence_gate_eval.py
   ```

2. **Bei Erfolg:** Vollständige FRANK Evaluation (300 Examples)

3. **Falls immer noch Probleme:**
   - Prüfen, ob das LLM überhaupt Evidence zurückgibt
   - Prompt verbessern (expliziter nach Evidence fragen)
   - Evidence-Erkennung weiter verbessern






