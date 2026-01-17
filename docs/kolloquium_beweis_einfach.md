# Beweis: Evidence-Gate im Factuality-Agent (für Kolloquium)

## E) Code-Beleg: Regel "ohne Beleg → unsicher"

**Datei:** `app/services/agents/factuality/claim_verifier.py`  
**Funktion:** `_apply_gate()` (Zeile 345-348)

```python
# Gate 2: incorrect ohne Evidence => uncertain
elif label_raw == "incorrect" and not selection.evidence_found:
    label_final = "uncertain"
    conf = min(conf, 0.5)
    gate_reason = "no_evidence"
```

**Erklärung:** Wenn das System "falsch" erkennt, aber keinen Beleg im Artikel findet, wird automatisch auf "unsicher" umgestellt.

---

## A) 2 Mini-Beispiele

### Beispiel 1: Klar falsch + Beleg vorhanden

```json
{
  "article_text": "Berlin ist die Hauptstadt von Deutschland. Die Stadt hat über 3,7 Millionen Einwohner.",
  "summary_text": "Paris ist die Hauptstadt von Deutschland.",
  "dataset": "demo",
  "llm_model": "gpt-4o-mini"
}
```

**Erwarteter Output:** `factuality.issue_spans[0].verdict == "incorrect"` UND `evidence_found == true`

---

### Beispiel 2: Behauptung ohne Beleg

```json
{
  "article_text": "Berlin ist eine große Stadt in Deutschland. Die Stadt wurde 1990 wieder vereint.",
  "summary_text": "Berlin hat genau 4 Millionen Einwohner.",
  "dataset": "demo",
  "llm_model": "gpt-4o-mini"
}
```

**Erwarteter Output:** `factuality.issue_spans[0].verdict == "uncertain"` (NICHT "incorrect") UND `evidence_found == false`

---

## B) Beispiel mit Markierungen (issue_spans)

```json
{
  "article_text": "Berlin ist die Hauptstadt von Deutschland. Die Stadt hat über 3,7 Millionen Einwohner.",
  "summary_text": "Paris ist die Hauptstadt von Deutschland. Die Stadt hat 4 Millionen Einwohner.",
  "dataset": "demo",
  "llm_model": "gpt-4o-mini"
}
```

**Erwarteter Output:** `factuality.issue_spans` enthält Objekte mit:
- `start_char`: 0 (Startposition im Text)
- `end_char`: 45 (Endposition im Text)
- `message`: Beschreibung des Problems
- `verdict`: "incorrect" oder "uncertain"

---

## C) Befehle zum Ausführen

### Option 1: Python-Script (empfohlen)

Erstelle Datei `test_beweis.py`:

```python
#!/usr/bin/env python3
import requests
import json

BASE_URL = "http://localhost:8000"

# Beispiel 1: Falsch + Beleg
print("\n" + "="*60)
print("Beispiel 1: Falsch + Beleg vorhanden")
print("="*60)
response1 = requests.post(f"{BASE_URL}/verify", json={
    "article_text": "Berlin ist die Hauptstadt von Deutschland. Die Stadt hat über 3,7 Millionen Einwohner.",
    "summary_text": "Paris ist die Hauptstadt von Deutschland.",
    "dataset": "demo",
    "llm_model": "gpt-4o-mini"
})
result1 = response1.json()
fact1 = result1["factuality"]
if fact1.get("issue_spans"):
    span1 = fact1["issue_spans"][0]
    print(f"✅ Verdict: {span1.get('verdict')}")
    print(f"✅ Evidence Found: {span1.get('evidence_found')}")
    print(f"✅ Message: {span1.get('message', '')[:80]}...")

# Beispiel 2: Ohne Beleg
print("\n" + "="*60)
print("Beispiel 2: Behauptung ohne Beleg")
print("="*60)
response2 = requests.post(f"{BASE_URL}/verify", json={
    "article_text": "Berlin ist eine große Stadt in Deutschland. Die Stadt wurde 1990 wieder vereint.",
    "summary_text": "Berlin hat genau 4 Millionen Einwohner.",
    "dataset": "demo",
    "llm_model": "gpt-4o-mini"
})
result2 = response2.json()
fact2 = result2["factuality"]
if fact2.get("issue_spans"):
    span2 = fact2["issue_spans"][0]
    print(f"✅ Verdict: {span2.get('verdict')}")
    print(f"✅ Evidence Found: {span2.get('evidence_found')}")
    print(f"✅ Message: {span2.get('message', '')[:80]}...")

# Beispiel 3: Markierungen
print("\n" + "="*60)
print("Beispiel 3: Markierungen (issue_spans)")
print("="*60)
response3 = requests.post(f"{BASE_URL}/verify", json={
    "article_text": "Berlin ist die Hauptstadt von Deutschland. Die Stadt hat über 3,7 Millionen Einwohner.",
    "summary_text": "Paris ist die Hauptstadt von Deutschland. Die Stadt hat 4 Millionen Einwohner.",
    "dataset": "demo",
    "llm_model": "gpt-4o-mini"
})
result3 = response3.json()
fact3 = result3["factuality"]
print(f"✅ Anzahl Issue Spans: {len(fact3.get('issue_spans', []))}")
for i, span in enumerate(fact3.get("issue_spans", [])[:2]):
    print(f"\n   Span {i+1}:")
    print(f"   - Start: {span.get('start_char')}, End: {span.get('end_char')}")
    print(f"   - Verdict: {span.get('verdict')}")
    print(f"   - Message: {span.get('message', '')[:60]}...")
```

**Ausführen:**
```bash
cd /Users/lisasimon/PycharmProjects/veri-api
source .venv/bin/activate
pip install requests  # falls nicht installiert
python3 test_beweis.py
```

### Option 2: curl (Alternative)

```bash
# Beispiel 1
curl -X POST "http://localhost:8000/verify" \
  -H "Content-Type: application/json" \
  -d '{"article_text":"Berlin ist die Hauptstadt von Deutschland. Die Stadt hat über 3,7 Millionen Einwohner.","summary_text":"Paris ist die Hauptstadt von Deutschland.","dataset":"demo","llm_model":"gpt-4o-mini"}' \
  | python3 -m json.tool | grep -A 10 "issue_spans"

# Beispiel 2
curl -X POST "http://localhost:8000/verify" \
  -H "Content-Type: application/json" \
  -d '{"article_text":"Berlin ist eine große Stadt in Deutschland. Die Stadt wurde 1990 wieder vereint.","summary_text":"Berlin hat genau 4 Millionen Einwohner.","dataset":"demo","llm_model":"gpt-4o-mini"}' \
  | python3 -m json.tool | grep -A 10 "issue_spans"
```

---

## D) Screenshot-Liste (max. 4)

### Screenshot 1: Swagger UI (/docs)
**Was:** Browser mit `http://localhost:8000/docs` geöffnet, `/verify` Endpoint sichtbar
**Wofür:** Zeigt, dass das System über REST-API zugänglich ist
**Woher:** Browser → `http://localhost:8000/docs`

### Screenshot 2: Response Beispiel 1 (falsch + Beleg)
**Was:** Terminal oder JSON-Viewer mit Response von Beispiel 1, fokussiert auf:
- `factuality.issue_spans[0].verdict: "incorrect"`
- `factuality.issue_spans[0].evidence_found: true`
- `factuality.details.claims[0].evidence_quote` (Text aus Artikel sichtbar)
**Wofür:** Zeigt, dass falsche Behauptungen nur mit Beleg als "falsch" markiert werden
**Woher:** Terminal (Python-Script Output) oder Swagger UI Response-Panel

### Screenshot 3: Response Beispiel 2 (ohne Beleg → unsicher)
**Was:** Terminal oder JSON-Viewer mit Response von Beispiel 2, fokussiert auf:
- `factuality.issue_spans[0].verdict: "uncertain"` (NICHT "incorrect")
- `factuality.issue_spans[0].evidence_found: false`
**Wofür:** Zeigt, dass Behauptungen ohne Beleg als "unsicher" markiert werden (nicht "falsch")
**Woher:** Terminal (Python-Script Output) oder Swagger UI Response-Panel

### Screenshot 4: Response mit issue_spans (Markierungen)
**Was:** Terminal oder JSON-Viewer mit Response von Beispiel 3, fokussiert auf:
- `factuality.issue_spans[0].start_char: 0` (konkrete Zahl)
- `factuality.issue_spans[0].end_char: 45` (konkrete Zahl)
- `factuality.issue_spans[0].message` (Beschreibung sichtbar)
**Wofür:** Zeigt, dass konkrete Textstellen als Char-Positionen markiert werden
**Woher:** Terminal (Python-Script Output) oder Swagger UI Response-Panel

---

**Tipp für Präsentation:** Screenshots 2-4 können auch kombiniert werden (alle 3 Responses in einem Screenshot, nebeneinander).





