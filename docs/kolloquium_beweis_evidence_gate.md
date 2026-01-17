# Beweis: Evidence-Gate im Factuality-Agent

## Evidence-Gate: Code-Beleg

**Datei:** `app/services/agents/factuality/claim_verifier.py`

**Funktion:** `_apply_gate()` (Zeilen 312-362)

**Kurze Erklärung:**
Die Gate-Logik stellt sicher, dass "incorrect" nur gesetzt wird, wenn `evidence_found == True` ist. Wenn das LLM "incorrect" ohne Evidence ausgibt, wird es automatisch zu "uncertain" downgraded (Zeile 345-348). Die Evidence-Validierung erfolgt in `_validate_evidence()` (Zeilen 221-293), die prüft, ob ein wörtliches Zitat aus dem Artikel als Substring in der ausgewählten Passage gefunden wurde.

**Span-Markierung:** `app/services/agents/factuality/factuality_agent.py`, Funktion `_build_issue_spans_from_claims()` (Zeilen 476-488) erstellt `IssueSpan`-Objekte mit `start_char` und `end_char` basierend auf der Claim-Position im Summary-Text.

---

## 3 Demo-Payloads

### Testfall A: Klar falsch + Evidenz vorhanden

```json
{
  "article_text": "Berlin ist die Hauptstadt von Deutschland. Die Stadt hat über 3,7 Millionen Einwohner und ist seit 1990 wieder vereint.",
  "summary_text": "Paris ist die Hauptstadt von Deutschland.",
  "dataset": "demo",
  "llm_model": "gpt-4o-mini"
}
```

**Erwartung:** `factuality.issue_spans[0].verdict == "incorrect"` UND `factuality.issue_spans[0].evidence_found == true` UND `factuality.details.claims[0].evidence_quote` enthält Text aus dem Artikel (z.B. "Berlin ist die Hauptstadt").

---

### Testfall B: Behauptung ohne Artikelbeleg

```json
{
  "article_text": "Berlin ist eine große Stadt in Deutschland. Die Stadt wurde 1990 wieder vereint.",
  "summary_text": "Berlin hat genau 4 Millionen Einwohner.",
  "dataset": "demo",
  "llm_model": "gpt-4o-mini"
}
```

**Erwartung:** `factuality.issue_spans[0].verdict == "uncertain"` (NICHT "incorrect") UND `factuality.issue_spans[0].evidence_found == false` ODER `factuality.details.claims[0].label == "uncertain"`.

---

### Testfall C: Span-Markierung sichtbar

```json
{
  "article_text": "Berlin ist die Hauptstadt von Deutschland. Die Stadt hat über 3,7 Millionen Einwohner.",
  "summary_text": "Paris ist die Hauptstadt von Deutschland. Die Stadt hat 4 Millionen Einwohner.",
  "dataset": "demo",
  "llm_model": "gpt-4o-mini"
}
```

**Erwartung:** `factuality.issue_spans` enthält mindestens ein Objekt mit `start_char` (Integer >= 0) UND `end_char` (Integer > start_char) UND `message` (String). Die Positionen markieren die fehlerhafte Textstelle im Summary.

---

## Demo-Commands

### Option 1: Python-Snippet (empfohlen)

```python
#!/usr/bin/env python3
"""Beweis: Evidence-Gate im Factuality-Agent - 3 Testfälle"""

import requests
import json

BASE_URL = "http://localhost:8000"

test_cases = {
    "A_falsch_mit_evidenz": {
        "article_text": "Berlin ist die Hauptstadt von Deutschland. Die Stadt hat über 3,7 Millionen Einwohner und ist seit 1990 wieder vereint.",
        "summary_text": "Paris ist die Hauptstadt von Deutschland.",
        "dataset": "demo",
        "llm_model": "gpt-4o-mini"
    },
    "B_ohne_evidenz": {
        "article_text": "Berlin ist eine große Stadt in Deutschland. Die Stadt wurde 1990 wieder vereint.",
        "summary_text": "Berlin hat genau 4 Millionen Einwohner.",
        "dataset": "demo",
        "llm_model": "gpt-4o-mini"
    },
    "C_span_markierung": {
        "article_text": "Berlin ist die Hauptstadt von Deutschland. Die Stadt hat über 3,7 Millionen Einwohner.",
        "summary_text": "Paris ist die Hauptstadt von Deutschland. Die Stadt hat 4 Millionen Einwohner.",
        "dataset": "demo",
        "llm_model": "gpt-4o-mini"
    }
}

for name, payload in test_cases.items():
    print(f"\n{'='*80}")
    print(f"Testfall: {name}")
    print(f"{'='*80}")
    
    response = requests.post(f"{BASE_URL}/verify", json=payload)
    result = response.json()
    
    factuality = result["factuality"]
    issue_spans = factuality.get("issue_spans", [])
    details = factuality.get("details", {})
    claims = details.get("claims", [])
    
    print(f"\n✅ Factuality Score: {factuality['score']:.3f}")
    print(f"✅ Issue Spans: {len(issue_spans)}")
    
    if issue_spans:
        span = issue_spans[0]
        print(f"   - Verdict: {span.get('verdict', 'N/A')}")
        print(f"   - Evidence Found: {span.get('evidence_found', 'N/A')}")
        print(f"   - Start/End: {span.get('start_char')}-{span.get('end_char')}")
        print(f"   - Message: {span.get('message', '')[:100]}...")
    
    if claims:
        claim = claims[0]
        print(f"\n✅ Erster Claim:")
        print(f"   - Label: {claim.get('label', 'N/A')}")
        print(f"   - Evidence Found: {claim.get('evidence_found', 'N/A')}")
        if claim.get('evidence_quote'):
            print(f"   - Evidence Quote: {claim['evidence_quote'][:80]}...")
    
    print(f"\n📄 Vollständige Response (JSON):")
    print(json.dumps(result, indent=2, ensure_ascii=False)[:500] + "...")
```

**Ausführen:**
```bash
cd /Users/lisasimon/PycharmProjects/veri-api
source .venv/bin/activate
python3 docs/kolloquium_beweis_evidence_gate.py
```

### Option 2: curl (Alternative)

```bash
# Testfall A
curl -X POST "http://localhost:8000/verify" \
  -H "Content-Type: application/json" \
  -d '{
    "article_text": "Berlin ist die Hauptstadt von Deutschland. Die Stadt hat über 3,7 Millionen Einwohner und ist seit 1990 wieder vereint.",
    "summary_text": "Paris ist die Hauptstadt von Deutschland.",
    "dataset": "demo",
    "llm_model": "gpt-4o-mini"
  }' | python3 -m json.tool

# Testfall B
curl -X POST "http://localhost:8000/verify" \
  -H "Content-Type: application/json" \
  -d '{
    "article_text": "Berlin ist eine große Stadt in Deutschland. Die Stadt wurde 1990 wieder vereint.",
    "summary_text": "Berlin hat genau 4 Millionen Einwohner.",
    "dataset": "demo",
    "llm_model": "gpt-4o-mini"
  }' | python3 -m json.tool

# Testfall C
curl -X POST "http://localhost:8000/verify" \
  -H "Content-Type: application/json" \
  -d '{
    "article_text": "Berlin ist die Hauptstadt von Deutschland. Die Stadt hat über 3,7 Millionen Einwohner.",
    "summary_text": "Paris ist die Hauptstadt von Deutschland. Die Stadt hat 4 Millionen Einwohner.",
    "dataset": "demo",
    "llm_model": "gpt-4o-mini"
  }' | python3 -m json.tool
```

---

## Beweis-Kriterien

### Testfall A: Klar falsch + Evidenz vorhanden
**Beweis-Kriterium:** Response enthält `factuality.issue_spans[0].verdict == "incorrect"` UND `factuality.issue_spans[0].evidence_found == true` UND `factuality.details.claims[0].evidence_quote` enthält wörtlichen Text aus dem Artikel (z.B. "Berlin ist die Hauptstadt").

### Testfall B: Behauptung ohne Artikelbeleg
**Beweis-Kriterium:** Response enthält `factuality.issue_spans[0].verdict == "uncertain"` (NICHT "incorrect") UND `factuality.issue_spans[0].evidence_found == false`, was zeigt, dass das System ohne Evidenz nicht "falsch" markiert, sondern "unsicher".

### Testfall C: Span-Markierung sichtbar
**Beweis-Kriterium:** Response enthält `factuality.issue_spans[0].start_char` (Integer >= 0) UND `factuality.issue_spans[0].end_char` (Integer > start_char), die die exakte Char-Position der fehlerhaften Textstelle im Summary markieren.

---

## Screenshot-Liste (Top 6)

### Screenshot 1: Code-Beleg - Gate-Logik
**Was:** `app/services/agents/factuality/claim_verifier.py`, Zeilen 345-348 (Gate 2: incorrect ohne Evidence → uncertain)
**Wofür:** Folie 1 - Zeigt, dass Evidence-Gate im Code implementiert ist
**Woher:** Code-Editor mit Syntax-Highlighting

### Screenshot 2: Code-Beleg - Evidence-Validierung
**Was:** `app/services/agents/factuality/claim_verifier.py`, Zeilen 267-272 (Quote-Matching: quote muss Substring der Passage sein)
**Wofür:** Folie 1 - Zeigt, wie Evidence validiert wird
**Woher:** Code-Editor mit Syntax-Highlighting

### Screenshot 3: Testfall A - Response (incorrect + evidence_found=true)
**Was:** Terminal-Output oder JSON-Viewer mit Response von Testfall A, fokussiert auf `factuality.issue_spans[0]` mit `verdict: "incorrect"` und `evidence_found: true` sowie `evidence_quote` sichtbar
**Wofür:** Folie 2 - Zeigt, dass falsche Behauptungen nur mit Evidenz als "incorrect" markiert werden
**Woher:** Terminal (Python-Script Output) oder JSON-Viewer (curl Response)

### Screenshot 4: Testfall B - Response (uncertain ohne evidence_found)
**Was:** Terminal-Output oder JSON-Viewer mit Response von Testfall B, fokussiert auf `factuality.issue_spans[0]` mit `verdict: "uncertain"` und `evidence_found: false`
**Wofür:** Folie 2 - Zeigt, dass Behauptungen ohne Evidenz als "uncertain" markiert werden (nicht "incorrect")
**Woher:** Terminal (Python-Script Output) oder JSON-Viewer (curl Response)

### Screenshot 5: Testfall C - Span-Markierung (start_char/end_char)
**Was:** Terminal-Output oder JSON-Viewer mit Response von Testfall C, fokussiert auf `factuality.issue_spans[0]` mit `start_char: X` und `end_char: Y` (konkrete Zahlen) sowie `message` sichtbar
**Wofür:** Folie 2 - Zeigt, dass konkrete Textstellen als Char-Positionen markiert werden
**Woher:** Terminal (Python-Script Output) oder JSON-Viewer (curl Response)

### Screenshot 6: Swagger UI - Request/Response Side-by-Side
**Was:** Swagger UI (`http://localhost:8000/docs`) mit ausgefülltem `/verify` Request (Testfall A) oben und Response unten, beide sichtbar
**Wofür:** Folie 1 - Zeigt, dass das System über REST-API zugänglich ist und live getestet werden kann
**Woher:** Browser (Swagger UI)

---

**Hinweis:** Für die Präsentation können Screenshots 3-5 auch kombiniert werden (alle 3 Responses in einem Screenshot, nebeneinander oder untereinander).





