
# M8 – Readability-Agent

## Ziel

In M8 wurde ein **Readability-Agent** umgesetzt, der die **Lesbarkeit von Zusammenfassungen** bewertet (Klarheit, Stil, Verständlichkeit) und das Ergebnis im **einheitlichen `AgentResult`-Format** zurückgibt. Der Agent ist **voll in die bestehende Verifikationspipeline integriert** und seine Ergebnisse werden sowohl in **Postgres** (für Runs/Analytics) als auch optional in **Neo4j** (für graphbasierte Nachvollziehbarkeit) persistiert.

---

## Aufgaben (und Umsetzung)

### 1) Kriterien für Lesbarkeit definieren
Readability bewertet primär sprachliche Verständlichkeit, u.a.:

- Klarheit und Verständlichkeit des Satzbaus
- Satzlänge und Verschachtelung
- Zeichensetzungs-Overload (Klammern/Kommas)
- unnötige Komplexität / schwer lesbarer Stil

**Wichtig:** Readability ist von **Coherence** abgegrenzt. Coherence bewertet logische Struktur und Widersprüche, nicht sprachliche Komplexität.

---

### 2) `ReadabilityAgent` implementieren (LLM-basiert)
Der Agent ist implementiert als Wrapper um einen LLM-Evaluator (vgl. `app/services/agents/readability/readability_agent.py:27-284`):

- `ReadabilityAgent` ruft intern `LLMReadabilityEvaluator` auf (vgl. `app/services/agents/readability/readability_verifier.py`)
- Unterstützt zwei Prompt-Versionen:
  - **v1** (Default): Score 0.0-1.0 (vgl. `app/services/agents/readability/readability_verifier.py:159-206`)
  - **v2**: Rubrik-basiertes 1-5 Rating (`score_raw_1_to_5` als Integer), wird intern zu 0-1 normalisiert: `score = clamp01((score_raw_1_to_5 - 1) / 4)` (vgl. `app/services/agents/readability/readability_verifier.py:81-86, 208-254`)
  - Constraint: Wenn `score_raw_1_to_5 <= 2`, dann min 1 issue (vgl. `app/services/agents/readability/readability_verifier.py:269`)
  - Raw score wird zusätzlich im Ergebnis gespeichert (vgl. `app/services/agents/readability/readability_verifier.py:352-356`)
- Ausgabe ist pipeline-kompatibel und deterministisch strukturierbar (Score + Issues + Erklärung)

---

### 3) AgentResult-Struktur nutzen (`score`, `errors`, `explanation`)
Der Output folgt dem projekteinheitlichen Format:

- `score`: normalisierter Readability-Score (typisch 0.0–1.0)
- `explanation`: kurze Begründung in Textform
- `issue_spans`: lokalisierte Problemstellen im Summary-Text (Issue-Typen: LONG_SENTENCE, COMPLEX_NESTING, PUNCTUATION_OVERLOAD, HARD_TO_PARSE) (vgl. `app/services/agents/readability/readability_verifier.py:255`)
- `details`: strukturierte Zusatzdaten (z.B. Issue-Liste, Rohsignale)

**Span-Mapping:** Issues enthalten einen `summary_span`, der per `find()` im Summary-Text auf `(start_char, end_char)` gemappt wird, damit die Problemstellen zuverlässig referenzierbar sind.

---

### 4) Integration in Pipeline, Postgres, Neo4j

#### Pipeline / API
- `/verify` führt Readability zusammen mit den anderen Agenten aus (end-to-end)
- `ReadabilityAgent` ist Bestandteil der Pipeline und liefert ein `AgentResult`, das wie alle anderen Dimensionen behandelt wird

#### Postgres Persistenz
- Ergebnisse landen in `verification_results`
- Speicherung pro Dimension (z.B. `dimension="readability"`) mit:
  - `score`
  - `details` als JSONB (inkl. Issue-Infos)

#### Neo4j Persistenz (optional)
Über `write_verification_graph(...)` werden Runs als Graph abgelegt:

- `(:Article)-[:HAS_SUMMARY]->(:Summary)`
- `(:Summary)-[:HAS_METRIC]->(:Metric {run_id, dimension, score})`
- optional: `(:Metric)-[:HAS_ISSUE_SPAN]->(:IssueSpan:Error {run_id, summary_id, dimension, span_index, message, severity, start_char, end_char})` (vgl. `app/db/neo4j/graph_persistence.py:108-134`)

Damit ist Readability nicht nur “eine Zahl”, sondern auch **textuell verortet** und später im Graph analysierbar.

---

### 5) Unit-Tests + Beispielcases
- Tests laufen erfolgreich im Testmodus:
  - `TEST_MODE=1 pytest -q` → **11 passed**
- Smoke-Tests über `/verify`:
  - **good:** kurze, klare Summary → hoher Score, keine Spans
  - **bad:** extrem langer/verschachtelter Satz → niedriger Score, mehrere Spans (z.B. LONG_SENTENCE, COMPLEX_NESTING)

---

## Quickstart (Smoke Test)

```bash
curl -s -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "manual",
    "article_text": "Kurzer Artikeltext. Er dient nur als Kontext.",
    "summary_text": "Das ist ein unfassbar langer Satz, der immer weitergeht ...",
    "meta": {"test_case":"m8_readability_bad"}
  }' | python -m json.tool
````

---

## Nächste sinnvolle Schritte (nach M8)

* Readability-Evaluation auf SummEval in größeren Runs durchführen (z.B. 200/500/1700 mit Cache)
* Score-Normalisierung (z.B. SummEval 1–5 → 0–1) sauber dokumentieren
* Coherence-Prompt so schärfen, dass Readability-Probleme nicht fälschlich als Kohärenzfehler gewertet werden


---

