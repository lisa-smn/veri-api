# Evidence Retrieval Verbesserungen (Schritt 3)

## Problem
Nach dem ersten Test:
- **Recall: 0.000** ❌ (keine echten Fehler erkannt)
- **Specificity: 1.000** ✅ (keine False Positives)
- **Alle Claims werden zu "uncertain"** (50/50)

**Ursache:** Evidence wird nicht gefunden, weil:
1. Evidence muss wörtlich im Kontext vorkommen (zu strikt)
2. Coverage-Prüfung ist zu strikt
3. Evidence Retrieval findet möglicherweise zu wenig

## Implementierte Verbesserungen

### 1. Fuzzy Matching für Evidence ✅
- **Vorher:** Evidence muss wörtlich im Kontext vorkommen (`if s in context`)
- **Jetzt:** Fuzzy Matching mit Token-Overlap (mindestens 40% Overlap)
- **Methode:** `_fuzzy_match_evidence()` findet ähnlichste Sätze im Kontext

### 2. Coverage-Prüfung gelockert ✅
- **Vorher:** `evidence_found = bool(valid_evidence and coverage_ok)` (Coverage als harter Filter)
- **Jetzt:** `evidence_found = bool(valid_evidence)` (Coverage nur als Warnung, nicht als Filter)
- **Effekt:** Wenn Evidence gefunden wurde, zählt es als "evidence_found", auch wenn Coverage nicht perfekt ist

### 3. Evidence Retrieval Parameter verbessert ✅
- `top_k_sentences`: 8 → 15 (mehr Sätze)
- `max_context_chars`: 6000 → 8000 (mehr Kontext)
- `max_evidence`: 2 → 3 (mehr Evidence-Kandidaten)
- `require_evidence_for_correct`: True → False (lockern)
- `min_soft_token_coverage`: 0.5 → 0.3 (weniger strikt)

## Erwartete Verbesserungen

Nach diesen Änderungen sollte:
- **Mehr Evidence gefunden werden** (Fuzzy Matching + mehr Kontext)
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
   - Prompt verbessern (explizit nach Evidence fragen)
   - Coverage-Prüfung weiter lockern
   - Evidence-Erkennung im LLM-Output verbessern






