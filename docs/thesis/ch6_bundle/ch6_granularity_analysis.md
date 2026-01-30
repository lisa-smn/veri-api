# Granularitätsanalyse: Feingranular vs. Grobgranular

**Erstellt:** 2026-01-30  
**Zweck:** Dokumentation der Bewertungsgranularität des Systems

---

## 1) Datenstrukturen unterhalb der Summary

Das System verwendet eine **feingranulare Hierarchie**:

### Claim (atomare Ebene)
- **Definition:** Atomare, faktische Behauptung innerhalb eines Satzes
- **Struktur:** `app/services/agents/factuality/claim_models.py:16-50`
- **Felder:**
  - `id`: Eindeutige Claim-ID
  - `sentence_index`: Index des zugehörigen Satzes
  - `sentence`: Vollständiger Satz-Text
  - `text`: Claim-Text (kann Teil des Satzes sein)
  - `label`: `"correct"`, `"incorrect"`, `"uncertain"` (pro Claim)
  - `confidence`: Confidence-Wert [0, 1]
  - `error_type`: `"ENTITY"`, `"NUMBER"`, `"DATE"`, `"OTHER"` (nur bei incorrect)
  - `evidence_found`: Boolean (wurde Evidence gefunden?)
  - `evidence_quote`: Wörtlicher Textauszug aus Artikel
  - `explanation`: Begründung für das Urteil

### IssueSpan (Span-Ebene)
- **Definition:** Markiert problematische Textstellen im Summary (Char-Spans)
- **Struktur:** `app/models/pydantic.py:8-31`
- **Felder:**
  - `start_char`, `end_char`: Charakter-Positionen im Summary
  - `message`: Erklärung des Problems
  - `severity`: `"low"`, `"medium"`, `"high"`
  - `issue_type`: `"ENTITY"`, `"NUMBER"`, `"DATE"`, `"OTHER"`
  - `verdict`: `"incorrect"` oder `"uncertain"`
  - `confidence`: Confidence-Wert [0, 1]
  - `evidence_found`: Boolean
  - `evidence_quote`: Evidence-Passage aus Artikel

### SentenceResult (Satz-Ebene)
- **Definition:** Aggregiertes Ergebnis pro Satz (basierend auf Claims)
- **Struktur:** `app/services/agents/factuality/factuality_agent.py:15-22`
- **Felder:**
  - `sentence_index`: Index des Satzes
  - `sentence`: Satz-Text
  - `label`: `"correct"`, `"incorrect"`, `"uncertain"`, `"skipped"`
  - `confidence`: Confidence-Wert [0, 1]
  - `explanation`: Begründung

### AgentResult (Summary-Level)
- **Definition:** Finales Ergebnis für die gesamte Summary
- **Struktur:** `app/models/pydantic.py:34-55`
- **Felder:**
  - `score`: Gesamt-Score [0, 1]
  - `explanation`: Globale Erklärung
  - `issue_spans`: Liste von IssueSpan-Objekten
  - `details`: Zusatzinfos (Claims, SentenceResults, etc.)

---

## 2) Stellen, wo pro Claim/Satz/Span Urteile erzeugt werden

### Claim-Extraktion (pro Satz)
- **Datei:** `app/services/agents/factuality/claim_extractor.py`
- **Zeilen:** `40-51` (`extract_claims`)
- **Logik:** 
  - Summary wird in Sätze zerlegt (`factuality_agent.py:551-636`)
  - Pro Satz werden Claims extrahiert (LLM-basiert)
  - Ein Satz kann mehrere Claims enthalten

### Claim-Verifikation (pro Claim)
- **Datei:** `app/services/agents/factuality/claim_verifier.py`
- **Zeilen:** `75-100` (`verify`)
- **Logik:**
  - Jeder Claim wird einzeln gegen den Artikel verifiziert
  - Evidence Retrieval → LLM-Verifikation → Gate-Logik
  - Output: `label` (correct/incorrect/uncertain), `confidence`, `error_type`, `evidence_found`

### Sentence-Aggregation (pro Satz)
- **Datei:** `app/services/agents/factuality/factuality_agent.py`
- **Zeilen:** `327-376` (`_aggregate_sentence_results`)
- **Logik:**
  - Claims werden nach `sentence_index` gruppiert
  - Pro Satz wird das stärkste Label genommen:
    - Wenn `incorrect` Claims vorhanden: stärkster incorrect (höchste Confidence)
    - Sonst wenn `uncertain` Claims vorhanden: stärkster uncertain
    - Sonst: `correct` (niedrigste Confidence aller correct Claims)

### IssueSpan-Generierung (pro Satz, repräsentativ)
- **Datei:** `app/services/agents/factuality/factuality_agent.py`
- **Zeilen:** `416-436` (`_representative_claims_for_spans`), `438-508` (`_build_issue_spans_from_claims`)
- **Logik:**
  - Pro Satz wird maximal 1 repräsentativer Claim ausgewählt:
    - Wenn `incorrect` Claims vorhanden: stärkster incorrect
    - Sonst wenn `uncertain` Claims vorhanden: stärkster uncertain
  - Dieser Claim wird in ein `IssueSpan` mit Char-Positionen (`start_char`, `end_char`) konvertiert

---

## 3) Aggregationslogik: Claim/Span → Summary-Level

### Claims → SentenceResult (Satz-Level)
**Regel:** Pro Satz wird das stärkste Label aller Claims genommen (incorrect > uncertain > correct), wobei "stärkster" = höchste Confidence bedeutet.

**Code:** `app/services/agents/factuality/factuality_agent.py:327-376`

**Details:**
- Claims werden nach `sentence_index` gruppiert
- Wenn ein Satz mehrere Claims hat:
  - Wenn mindestens 1 `incorrect` Claim: Satz = `incorrect` (Confidence = max(incorrect_claims.confidence))
  - Sonst wenn mindestens 1 `uncertain` Claim: Satz = `uncertain` (Confidence = max(uncertain_claims.confidence))
  - Sonst: Satz = `correct` (Confidence = min(correct_claims.confidence))

### SentenceResult → Score (Summary-Level)
**Regel:** Score = (Anzahl correct Sätze + 0.5 × Anzahl uncertain Sätze) / Anzahl geprüfter Sätze

**Code:** `app/services/agents/factuality/factuality_agent.py:380-388`

**Formel:**
```python
score = (correct + UNCERTAIN_WEIGHT * uncertain) / total_checked
# wobei UNCERTAIN_WEIGHT = 0.5
```

**Details:**
- `skipped` Sätze werden nicht gezählt
- `correct` Sätze zählen voll (1.0)
- `uncertain` Sätze zählen halb (0.5)
- `incorrect` Sätze zählen nicht (0.0)

### Claims → IssueSpan (für Explainability)
**Regel:** Pro Satz wird maximal 1 repräsentativer Claim (stärkster incorrect oder uncertain) in ein IssueSpan mit Char-Positionen konvertiert.

**Code:** `app/services/agents/factuality/factuality_agent.py:416-508`

**Details:**
- Nur `incorrect` oder `uncertain` Claims werden zu IssueSpans
- Pro Satz maximal 1 IssueSpan (repräsentativer Claim)
- Char-Positionen werden durch Substring-Matching im Satz-Text lokalisiert

---

## 4) Beispiel aus Result-Datei

**Quelle:** `results/evaluation/runs/results/evidence_gate_test_count_as_error_examples.jsonl` (Zeile 1)

```json
{
  "example_id": "ex_0",
  "ground_truth": true,
  "prediction": true,
  "score": 0.0,
  "num_issues": 1,
  "effective_issues": 1.0,
  "weighted_score": null,
  "issue_spans": [
    {
      "start_char": 0,
      "end_char": 73,
      "message": "Satz 1: Claim 'the fbi has said it will help the san bernardino killer to access iphones' – Die Passage beschreibt, dass die FBI in einem anderen Fall in Arkansas helfen wird, nicht dass sie dem San Bernardino-Killer helfen werden, auf iPhones zuzugreifen.",
      "severity": "low",
      "issue_type": "OTHER",
      "confidence": 0.5,
      "mapping_confidence": 1.0,
      "evidence_found": true,
      "verdict": "incorrect"
    }
  ],
  "summary": "the fbi has said it will help the san bernardino killer to access iphones used by the san bernardino victims.",
  "meta": {
    "hash": "35933239",
    "model_name": "BERTS2S",
    "factuality": 0.0
  }
}
```

**Interpretation:**
- **Summary-Level:** `score=0.0` (kein correct Satz)
- **Satz-Level:** Satz 1 hat ein Problem (incorrect)
- **Claim-Level:** Ein Claim wurde extrahiert und als `incorrect` verifiziert
- **Span-Level:** IssueSpan markiert Charaktere 0-73 im Summary-Text
- **Evidence:** `evidence_found=true`, `evidence_quote` enthält die widerlegende Passage

**Vollständige Claim-Details** (in `AgentResult.details.claims`):
```json
{
  "id": "claim_0_0",
  "sentence_index": 0,
  "sentence": "the fbi has said it will help the san bernardino killer to access iphones used by the san bernardino victims.",
  "text": "the fbi has said it will help the san bernardino killer to access iphones",
  "label": "incorrect",
  "confidence": 0.5,
  "error_type": "OTHER",
  "explanation": "Die Passage beschreibt, dass die FBI in einem anderen Fall in Arkansas helfen wird...",
  "evidence_found": true,
  "evidence_quote": "...",
  "selected_evidence_index": 0,
  "retrieved_passages": [...],
  "retrieval_scores": [...]
}
```

---

## Zusammenfassung: Feingranular oder Grobgranular?

**Antwort: Das System ist FEINGRANULAR.**

### Bewertungsebenen:
1. **Claim-Ebene:** Atomare Claims werden extrahiert und einzeln verifiziert
2. **Satz-Ebene:** Claims werden zu Satz-Labels aggregiert
3. **Span-Ebene:** Problematische Stellen werden als Char-Spans markiert
4. **Summary-Ebene:** Satz-Labels werden zu einem Gesamt-Score aggregiert

### Speicherung:
- **Pro Claim:** Vollständige Claim-Details in `AgentResult.details.claims`
- **Pro Satz:** SentenceResult in `AgentResult.details.sentence_results`
- **Pro Span:** IssueSpan in `AgentResult.issue_spans`
- **Pro Summary:** Score und Explanation in `AgentResult`

### Aggregationsregeln:
- **Claims → Satz:** Stärkstes Label (incorrect > uncertain > correct)
- **Sätze → Score:** Gewichtete Summe (correct=1.0, uncertain=0.5, incorrect=0.0)
- **Claims → Spans:** Repräsentativer Claim pro Satz (stärkster incorrect/uncertain)
