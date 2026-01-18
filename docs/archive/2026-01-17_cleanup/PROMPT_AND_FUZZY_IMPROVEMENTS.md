# Prompt und Fuzzy Matching Verbesserungen

## Problem
Nach allen Verbesserungen bleibt das Problem:
- **Recall: 0.000** ❌ (keine echten Fehler erkannt)
- **Alle Claims werden zu "uncertain"** (50/50)

**Ursache:** Das LLM gibt möglicherweise keine Evidence zurück, oder die Evidence wird nicht gefunden.

## Implementierte Verbesserungen

### 1. Prompt verbessert ✅
- **Vorher:** "Wenn du keine passenden Zitate findest: evidence = []"
- **Jetzt:** "Suche AKTIV nach wörtlichen Zitaten im KONTEXT"
- **Jetzt:** "Kopiere die relevanten Sätze/Teilsätze WÖRTLICH aus dem KONTEXT"
- **Jetzt:** Explizite Anweisung: "Für 'incorrect': Gib das Zitat an, das dem Claim widerspricht"
- **Jetzt:** "Maximal 3 Zitate" (vorher 2)

**Effekt:** Das LLM wird ermutigt, aktiver nach Evidence zu suchen und wörtliche Zitate zu kopieren.

### 2. Fuzzy Matching gelockert ✅
- **Vorher:** `min_overlap_ratio = 0.4` (40% Token-Overlap)
- **Jetzt:** `min_overlap_ratio = 0.25` (25% Token-Overlap)

**Effekt:** Mehr Evidence wird gefunden, auch wenn nicht exakt übereinstimmend.

## Erwartete Verbesserungen

Nach diesen Änderungen sollte:
- **Mehr Evidence vom LLM zurückgegeben werden** (besserer Prompt)
- **Mehr Evidence gefunden werden** (lockeres Fuzzy Matching)
- **Mehr "incorrect" Claims durchkommen** (weil Evidence gefunden wird)
- **Recall steigen** (von 0.000 auf > 0.70)
- **Specificity hoch bleiben** (>= 0.20)

## Nächste Schritte

1. **Test nochmal laufen lassen:**
   ```bash
   python3 scripts/test_evidence_gate_eval.py
   ```

2. **Bei Erfolg:** Vollständige FRANK Evaluation (300 Examples)

3. **Falls immer noch Probleme:**
   - Prüfen, ob das LLM überhaupt "incorrect" zurückgibt
   - Evidence-Erkennung weiter verbessern
   - Möglicherweise LLM-Modell wechseln (gpt-4o statt gpt-4o-mini)






