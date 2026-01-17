# Explainability-Modul Audit

**Zweck:** Prüft, ob das Explainability-Modul seinen Zweck erfüllt: "Das Urteil nachvollziehbar machen."

---

## Was es tut / Was es nicht tut

### Was es tut

1. **Normalisiert Agent-Outputs:** Konvertiert `issue_spans` aus allen 3 Agenten in ein einheitliches `Finding`-Format (Dimension, Severity, Message, Span, Evidence).
2. **Dedupliziert & Clustert:** Überlappende Findings werden zusammengeführt (z.B. mehrere Spans im selben Satz).
3. **Rankt Findings:** Priorisierung nach `Severity × Dimension-Gewichtung × log(Span-Länge)`.
4. **Generiert Executive Summary:** Regelbasierte Zusammenfassung (3-6 Sätze) mit Anzahl Findings, Schwerpunkt-Dimension, Top-Spans.
5. **Strukturierte Ausgabe:** `ExplainabilityResult` mit `summary[]`, `findings[]`, `by_dimension{}`, `top_spans[]`, `stats{}`, `version`.

### Was es nicht tut

1. **Keine externe Evidence für Coherence:** Coherence-Findings haben keine Evidence-Quotes aus dem Artikel (nur für Factuality).
2. **Keine Claim-Verlinkung:** Findings referenzieren nicht explizit Claims (nur über `source.claim_id` in Evidence).
3. **Keine temporale Analyse:** Reihenfolge der Findings ist nach Ranking, nicht nach Auftreten im Text.

---

## Datenfluss

```
PipelineResult (factuality, coherence, readability)
  ↓
ExplainabilityService.build(pipeline_result, summary_text)
  ↓
  ├─ _normalize_factuality() → Findings aus issue_spans + details
  ├─ _normalize_generic() → Findings aus issue_spans (coherence/readability)
  ├─ _dedupe_and_cluster() → Entfernt Duplikate, clustert überlappende Spans
  ├─ _rank() → Sortiert nach Severity × Dimension × Span-Länge
  ├─ _top_spans() → Top-K wichtigste Spans
  ├─ _stats() → num_findings, coverage_chars, coverage_ratio
  └─ _executive_summary() → Regelbasierte Zusammenfassung
  ↓
ExplainabilityResult
```

**Entry Points:**
- `app/pipeline/verification_pipeline.py` (Zeile 58): `result.explainability = self.explainability_service.build(result, summary_text=summary)`
- `app/api/routes.py` (Zeile 30): `explainability=result.explainability` im `VerifyResponse`

**Persistenz:**
- `app/db/postgres/persistence.py` (Zeile 226): `store_verification_run()` speichert `explainability` als JSON in `explainability_reports` Tabelle.

---

## Beispiel 1: Klarer Faktenfehler

### Input
- **Artikel:** "Die Studie wurde 2023 veröffentlicht. 150 Teilnehmer nahmen teil."
- **Summary:** "Die Studie wurde 2022 veröffentlicht. 200 Teilnehmer nahmen teil."

### System-Output
- **Factuality Score:** 0.3 (niedrig, da 2 Fehler)
- **Issue Spans:**
  - `"2022"` (start_char: 25, end_char: 29) → `issue_type: "DATE"`, `severity: "high"`
  - `"200"` (start_char: 50, end_char: 53) → `issue_type: "NUMBER"`, `severity: "high"`

### Explainability-Auszug
```json
{
  "summary": [
    "In der Summary wurden 2 Findings identifiziert (2 high, 0 medium, 0 low).",
    "Der Schwerpunkt liegt bei **factuality** (alle Findings in dieser Dimension)."
  ],
  "findings": [
    {
      "id": "factuality_high_DATE_25_29_...",
      "dimension": "factuality",
      "severity": "high",
      "message": "Datum '2022' widerspricht Artikel ('2023').",
      "span": {"start_char": 25, "end_char": 29, "text": "2022"},
      "evidence": [
        {"kind": "quote", "source": "agent:factuality", "quote": "Die Studie wurde 2023 veröffentlicht."}
      ],
      "recommendation": "Datum mit dem Artikel abgleichen."
    },
    {
      "id": "factuality_high_NUMBER_50_53_...",
      "dimension": "factuality",
      "severity": "high",
      "message": "Zahl '200' widerspricht Artikel ('150').",
      "span": {"start_char": 50, "end_char": 53, "text": "200"},
      "evidence": [
        {"kind": "quote", "source": "agent:factuality", "quote": "150 Teilnehmer nahmen teil."}
      ],
      "recommendation": "Zahlen mit dem Artikel abgleichen."
    }
  ],
  "top_spans": [
    {"span": {"start_char": 25, "end_char": 29}, "dimension": "factuality", "severity": "high", "rank_score": 8.5}
  ],
  "stats": {
    "num_findings": 2,
    "num_high_severity": 2,
    "coverage_chars": 8,
    "coverage_ratio": 0.15
  }
}
```

### So liest man es
1. **Executive Summary:** 2 Findings, beide high-severity, alle in Factuality.
2. **Top-Spans:** Wichtigste Stelle ist "2022" (Datum-Fehler).
3. **Evidence:** Jedes Finding hat ein Evidence-Quote aus dem Artikel, das den Widerspruch belegt.

---

## Beispiel 2: Klare Inkohärenz

### Input
- **Artikel:** "Die Studie zeigt positive Effekte. Die Teilnehmer berichteten Verbesserungen."
- **Summary:** "Die Studie zeigt positive Effekte. Die Teilnehmer berichteten keine Verbesserungen."

### System-Output
- **Coherence Score:** 0.4 (niedrig, da Widerspruch)
- **Issue Spans:**
  - `"Die Teilnehmer berichteten keine Verbesserungen."` (start_char: 45, end_char: 95) → `issue_type: "CONTRADICTION"`, `severity: "high"`

### Explainability-Auszug
```json
{
  "summary": [
    "In der Summary wurde 1 Finding identifiziert (1 high, 0 medium, 0 low).",
    "Der Schwerpunkt liegt bei **coherence** (alle Findings in dieser Dimension)."
  ],
  "findings": [
    {
      "id": "coherence_high_CONTRADICTION_45_95_...",
      "dimension": "coherence",
      "severity": "high",
      "message": "Widerspruch: 'positive Effekte' vs. 'keine Verbesserungen'.",
      "span": {"start_char": 45, "end_char": 95, "text": "Die Teilnehmer berichteten keine Verbesserungen."},
      "evidence": [
        {"kind": "raw", "source": "agent:coherence", "data": {"issue_span": {...}}}
      ],
      "recommendation": "Widersprüchliche Aussagen auflösen."
    }
  ],
  "top_spans": [
    {"span": {"start_char": 45, "end_char": 95}, "dimension": "coherence", "severity": "high", "rank_score": 7.2}
  ],
  "stats": {
    "num_findings": 1,
    "num_high_severity": 1,
    "coverage_chars": 50,
    "coverage_ratio": 0.52
  }
}
```

### So liest man es
1. **Executive Summary:** 1 Finding, high-severity, in Coherence.
2. **Top-Spans:** Wichtigste Stelle ist der Widerspruch-Satz.
3. **Evidence:** Keine externen Evidence-Quotes (nur Raw-Daten vom Agent), da Coherence keine Artikel-Evidence nutzt.

---

## Gap-Analyse

### Was fehlt oder ist unvollständig

1. **Evidence-Spans nur für Factuality vollständig:**
   - Factuality: Evidence-Quotes aus dem Artikel vorhanden.
   - Coherence: Keine externen Evidence-Quotes (nur interne Logik-Analyse).
   - Readability: Keine Evidence (nur strukturelle Analyse).

2. **Claim-Verlinkung nicht explizit:**
   - Findings referenzieren Claims nur über `source.claim_id` in Evidence, nicht als direkte Referenz.

3. **Temporale Analyse fehlt:**
   - Findings sind nach Ranking sortiert, nicht nach Auftreten im Text (könnte für "Story-Flow" nützlich sein).

---

## Nächste Verbesserungen (optional)

1. **Evidence für Coherence:** Externe Belege aus dem Artikel für Widersprüche (z.B. "Artikel sagt X, Summary sagt Y").
2. **Claim-Verlinkung:** Direkte Referenz von Findings zu Claims (z.B. `finding.claim_id`).
3. **Temporale Sortierung:** Optionale Sortierung nach `span.start_char` statt nur Ranking.

---

**Vollständige Implementierung:** `app/services/explainability/explainability_service.py`  
**Datenmodelle:** `app/services/explainability/explainability_models.py`

