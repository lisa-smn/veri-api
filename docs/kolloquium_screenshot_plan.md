# Screenshot-Plan für Kolloquium

## Schritt 0: Setup

**Startbefehl (empfohlen):**
```bash
cd /Users/lisasimon/PycharmProjects/veri-api
docker-compose up
```

**Alternative (ohne Docker):**
```bash
cd /Users/lisasimon/PycharmProjects/veri-api
source .venv/bin/activate
uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload
```

**Standard-URL:**
```
http://localhost:8000
```

**Prüfen ob Server läuft:**
```bash
curl http://localhost:8000/health
# Erwartet: {"status":"ok"}
```

---

## Screenshot-Checkliste

### Screenshot 01: API-Dokumentation (Swagger UI)

**Wo klicken/was ausführen:**
1. Browser öffnen
2. URL eingeben: `http://localhost:8000/docs`
3. Warten bis Swagger UI geladen ist

**Was muss sichtbar sein:**
- Linke Sidebar: `/verify` Endpoint (POST) sichtbar
- `/verify` Endpoint expandiert (auf "Try it out" klicken)
- Request Schema sichtbar: `article_text`, `summary_text`, `dataset`, `llm_model`
- Response Schema sichtbar: `factuality`, `coherence`, `readability`, `overall_score`, `explainability`

**Dateiname:**
```
01_docs_verify.png
```

---

### Screenshot 02: Input-Payload

**Wo klicken/was ausführen:**
1. In Swagger UI (`http://localhost:8000/docs`)
2. `/verify` Endpoint → "Try it out" klicken
3. Request Body ausfüllen mit folgendem JSON:

```json
{
  "article_text": "Berlin ist die Hauptstadt von Deutschland. Die Stadt hat über 3,7 Millionen Einwohner und ist seit 1990 wieder vereint.",
  "summary_text": "Paris ist die Hauptstadt von Deutschland.",
  "dataset": "demo",
  "llm_model": "gpt-4o-mini"
}
```

**Was muss sichtbar sein:**
- Request Body Editor mit ausgefülltem JSON
- `article_text` sichtbar (2-3 Sätze)
- `summary_text` sichtbar (1-2 Sätze)
- `dataset` und `llm_model` sichtbar

**Dateiname:**
```
02_input_payload.png
```

**Alternative (Terminal):**
```bash
# Payload in Datei speichern
cat > /tmp/payload.json << 'EOF'
{
  "article_text": "Berlin ist die Hauptstadt von Deutschland. Die Stadt hat über 3,7 Millionen Einwohner und ist seit 1990 wieder vereint.",
  "summary_text": "Paris ist die Hauptstadt von Deutschland.",
  "dataset": "demo",
  "llm_model": "gpt-4o-mini"
}
EOF

# Screenshot vom Terminal mit cat payload.json
cat /tmp/payload.json
```

---

### Screenshot 03: Output (Scores + issue_spans)

**Wo klicken/was ausführen:**

**Option A: Swagger UI (empfohlen)**
1. In Swagger UI, nachdem Request gesendet wurde
2. Response-Panel scrollen bis zu `factuality`, `coherence`, `readability`
3. `issue_spans` Array expandieren

**Option B: Terminal mit curl + jq**
```bash
curl -X POST "http://localhost:8000/verify" \
  -H "Content-Type: application/json" \
  -d '{
    "article_text": "Berlin ist die Hauptstadt von Deutschland. Die Stadt hat über 3,7 Millionen Einwohner und ist seit 1990 wieder vereint.",
    "summary_text": "Paris ist die Hauptstadt von Deutschland.",
    "dataset": "demo",
    "llm_model": "gpt-4o-mini"
  }' | jq '{
    overall_score,
    factuality: {score: .factuality.score, issue_spans: .factuality.issue_spans},
    coherence: {score: .coherence.score, issue_spans: .coherence.issue_spans},
    readability: {score: .readability.score, issue_spans: .readability.issue_spans}
  }'
```

**Was muss sichtbar sein:**
- `overall_score`: Zahl zwischen 0.0 und 1.0
- `factuality.score`: Zahl (z.B. 0.3)
- `coherence.score`: Zahl (z.B. 0.8)
- `readability.score`: Zahl (z.B. 0.9)
- `factuality.issue_spans[0]`: Objekt mit `start_char`, `end_char`, `verdict`, `message`, `severity`

**Dateiname:**
```
03_output_scores_spans.png
```

---

### Screenshot 04: Erklärungen (Explainability ODER Details)

**Wo klicken/was ausführen:**

**Option A: Swagger UI - factuality.details (empfohlen, funktioniert immer)**
1. In Swagger UI Response-Panel
2. Zu `factuality` scrollen
3. `factuality.details` expandieren
4. `factuality.details.claims[0]` expandieren
5. Zeige: `explanation`, `evidence_quote`, `label`, `evidence_found`

**Option B: Terminal mit jq (Details der Claims)**
```bash
curl -X POST "http://localhost:8000/verify" \
  -H "Content-Type: application/json" \
  -d '{
    "article_text": "Berlin ist die Hauptstadt von Deutschland. Die Stadt hat über 3,7 Millionen Einwohner und ist seit 1990 wieder vereint.",
    "summary_text": "Paris ist die Hauptstadt von Deutschland.",
    "dataset": "demo",
    "llm_model": "gpt-4o-mini"
  }' | jq '{
    factuality_explanation: .factuality.explanation,
    claims: [.factuality.details.claims[0:2] | {
      label,
      explanation,
      evidence_quote,
      evidence_found
    }]
  }'
```

**Was muss sichtbar sein:**
- `factuality.explanation`: String mit globaler Begründung (z.B. "Score 0.30. Es wurden faktische Inkonsistenzen erkannt...")
- `factuality.details.claims[0].explanation`: Claim-spezifische Erklärung (z.B. "Die Passage widerspricht dem Claim...")
- `factuality.details.claims[0].evidence_quote`: Wörtliches Zitat aus dem Artikel (z.B. "Berlin ist die Hauptstadt von Deutschland")
- `factuality.details.claims[0].label`: "incorrect" oder "uncertain"

**Alternative (falls explainability vorhanden):**
- `explainability.summary`: Array mit Executive Summary
- `explainability.stats.num_findings`: Anzahl der Findings
- `explainability.findings[0].message`: Erste Finding-Beschreibung

**Dateiname:**
```
04_output_explanations.png
```

---

### Screenshot 05 (Bonus): Evidence-Gate Vergleich

**Wo klicken/was ausführen:**

**Terminal mit zwei Requests:**

**Beispiel 1: Falsch + Beleg vorhanden**
```bash
curl -X POST "http://localhost:8000/verify" \
  -H "Content-Type: application/json" \
  -d '{
    "article_text": "Berlin ist die Hauptstadt von Deutschland. Die Stadt hat über 3,7 Millionen Einwohner.",
    "summary_text": "Paris ist die Hauptstadt von Deutschland.",
    "dataset": "demo",
    "llm_model": "gpt-4o-mini"
  }' | jq '.factuality.issue_spans[0] | {verdict, evidence_found, message}'
```

**Beispiel 2: Ohne Beleg**
```bash
curl -X POST "http://localhost:8000/verify" \
  -H "Content-Type: application/json" \
  -d '{
    "article_text": "Berlin ist eine große Stadt in Deutschland. Die Stadt wurde 1990 wieder vereint.",
    "summary_text": "Berlin hat genau 4 Millionen Einwohner.",
    "dataset": "demo",
    "llm_model": "gpt-4o-mini"
  }' | jq '.factuality.issue_spans[0] | {verdict, evidence_found, message}'
```

**Was muss sichtbar sein:**
- **Beispiel 1:** `verdict: "incorrect"`, `evidence_found: true`
- **Beispiel 2:** `verdict: "uncertain"`, `evidence_found: false`
- Beide Outputs nebeneinander im Terminal (oder zwei separate Screenshots)

**Dateiname:**
```
05_evidence_gate_comparison.png
```

---

## Copy-Paste Commands

### Kompletter Test-Workflow (Python)

```python
#!/usr/bin/env python3
import requests
import json

BASE_URL = "http://localhost:8000"

payload = {
    "article_text": "Berlin ist die Hauptstadt von Deutschland. Die Stadt hat über 3,7 Millionen Einwohner und ist seit 1990 wieder vereint.",
    "summary_text": "Paris ist die Hauptstadt von Deutschland.",
    "dataset": "demo",
    "llm_model": "gpt-4o-mini"
}

response = requests.post(f"{BASE_URL}/verify", json=payload, timeout=60)
result = response.json()

print("="*60)
print("SCORES:")
print(f"  Overall: {result['overall_score']:.3f}")
print(f"  Factuality: {result['factuality']['score']:.3f}")
print(f"  Coherence: {result['coherence']['score']:.3f}")
print(f"  Readability: {result['readability']['score']:.3f}")

print("\n" + "="*60)
print("ISSUE SPANS (Factuality):")
for i, span in enumerate(result['factuality']['issue_spans'][:2]):
    print(f"  Span {i+1}:")
    print(f"    Start: {span['start_char']}, End: {span['end_char']}")
    print(f"    Verdict: {span['verdict']}")
    print(f"    Evidence Found: {span['evidence_found']}")
    print(f"    Message: {span['message'][:60]}...")

if result.get('explainability'):
    print("\n" + "="*60)
    print("EXPLAINABILITY:")
    print(f"  Summary: {result['explainability']['summary'][0] if result['explainability']['summary'] else 'N/A'}")
    print(f"  Findings: {result['explainability']['stats']['num_findings']}")
    print(f"  High Severity: {result['explainability']['stats']['num_high_severity']}")
```

**Speichern als:** `demo_request.py`

**Ausführen:**
```bash
python3 demo_request.py
```

### Kompletter Test-Workflow (curl + jq)

```bash
# Einzelner Request mit schöner Formatierung
curl -X POST "http://localhost:8000/verify" \
  -H "Content-Type: application/json" \
  -d '{
    "article_text": "Berlin ist die Hauptstadt von Deutschland. Die Stadt hat über 3,7 Millionen Einwohner und ist seit 1990 wieder vereint.",
    "summary_text": "Paris ist die Hauptstadt von Deutschland.",
    "dataset": "demo",
    "llm_model": "gpt-4o-mini"
  }' | jq '{
    overall_score,
    factuality: {
      score: .factuality.score,
      issue_spans: [.factuality.issue_spans[0] | {start_char, end_char, verdict, evidence_found, message}]
    },
    coherence: {score: .coherence.score},
    readability: {score: .readability.score},
    explainability: {
      summary: .explainability.summary,
      stats: .explainability.stats,
      top_findings: [.explainability.findings[0:2] | {dimension, severity, message}]
    }
  }'
```

---

## Zusammenfassung

**Wenn diese 4 Screenshots (01-04) vorhanden sind, ist der Beweis vollständig:**
1. ✅ API existiert und ist dokumentiert (Swagger)
2. ✅ Input ist klar (Artikel + Summary als JSON)
3. ✅ Output zeigt 3 Dimensionen (Scores) + markierte Textstellen (issue_spans)
4. ✅ Erklärungen sichtbar (`factuality.explanation` + `factuality.details.claims[].explanation` + `evidence_quote`)

**Hinweis zu Screenshot 04:** Verwenden Sie `factuality.details.claims[]` für Erklärungen - das ist immer vorhanden und zeigt Claim-spezifische Begründungen mit Evidence-Quotes.

**Optional:** Screenshot 05 zeigt zusätzlich das Evidence-Gate (nur "falsch" mit Beleg, sonst "unsicher").

