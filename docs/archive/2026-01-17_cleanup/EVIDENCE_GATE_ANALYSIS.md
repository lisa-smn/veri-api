# Evidence-Gate Test Ergebnisse - Analyse

## Ergebnisse (50 FRANK Examples)

### Metrics
- **Recall:** 1.000 ✅ (perfekt!)
- **Specificity:** 0.000 ❌ (katastrophal - alle Negativen werden als Positiv klassifiziert)
- **Precision:** 0.940
- **F1:** 0.969
- **Balanced Accuracy:** 0.500

### Confusion Matrix
- **TP:** 47
- **TN:** 0 (alle 3 negativen Beispiele wurden als FP klassifiziert)
- **FP:** 3
- **FN:** 0

### Evidence-Gate Statistics
- **Incorrect ohne Evidence:** 0 ✅ (Evidence-Gate funktioniert!)
- **Incorrect mit Evidence:** 0
- **Uncertain:** 50 (alle Claims werden zu "uncertain")

## Problem-Analyse

### 1. Evidence-Gate funktioniert technisch ✅
- Keine "incorrect" Claims ohne Evidence
- Safety-Downgrade funktioniert

### 2. Alle Claims werden zu "uncertain" ❌
**Ursachen:**
1. **`require_evidence_for_correct=True`** - Auch "correct" Claims werden zu "uncertain" ohne Evidence
2. **Evidence-Coverage zu strikt** - `_evidence_covers_claim` prüft:
   - Hard units (Zahlen, Entities, Daten) müssen im Evidence vorkommen
   - Soft tokens müssen zu mindestens 50% (`min_soft_token_coverage=0.5`) vorkommen
3. **Evidence Retrieval könnte zu schlecht sein** - `_select_context` holt nur `top_k=8` Sätze

### 3. "Uncertain" Issues werden als "has_error" gezählt ❌
- Test-Script zählt alle IssueSpans als "has_error" (`len(agent_result.issue_spans) >= 1`)
- "Uncertain" Issues haben `severity="low"`, werden aber trotzdem gezählt
- Das führt zu vielen False Positives

## Lösungsansätze

### Priorität 1: Evidence Retrieval verbessern (Schritt 3)
- Mehr Sätze holen (`top_k` erhöhen)
- Besseres Scoring (BM25-ähnlich, TF-IDF)
- Explizite Evidence-Erkennung im LLM-Output

### Priorität 2: Test-Script anpassen
- "Uncertain" Issues sollten nicht automatisch als "has_error" gezählt werden
- Nur "incorrect" Issues (oder `severity != "low"`) als "has_error" zählen
- Oder: `uncertainty_policy=non_error` im Test-Script verwenden

### Priorität 3: `require_evidence_for_correct` lockern
- Nur für "incorrect" verwenden, nicht für "correct"
- Oder: `require_evidence_for_correct=False` für weniger strikte Prüfung

## Nächste Schritte

1. **Evidence Retrieval verbessern** (Schritt 3)
   - Top-k erhöhen (z.B. 15 statt 8)
   - BM25-ähnliches Scoring
   - Explizite Evidence-Erkennung

2. **Test-Script anpassen**
   - Nur "incorrect" Issues als "has_error" zählen
   - Oder: `severity != "low"` als Filter

3. **Vollständige FRANK Evaluation**
   - Nach Schritt 3: Vollständiger FRANK Run (300 Examples)
   - Prüfe: Specificity >= 0.20, Recall >= 0.90






