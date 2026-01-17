# Explainability-Modul: Specification

**Datum:** 2026-01-17  
**Version:** m9_v1  
**Zweck:** Definition der Contracts, Aggregationsregeln und Determinismus-Anforderungen

---

## Input Contract

### Erwartete Agent-Outputs

Das Explainability-Modul erwartet ein `PipelineResult`-ähnliches Objekt mit:

- **factuality**: `AgentResult` (oder dict) mit:
  - `issue_spans`: Liste von `IssueSpan`-Objekten (kanonisch)
    - `start_char`: int (Start-Position im Text)
    - `end_char`: int (End-Position im Text)
    - `message`: str (Beschreibung des Problems)
    - `severity`: str ("low", "medium", "high") oder int/float
    - `issue_type`: str (optional, z.B. "NUMBER", "DATE", "ENTITY")
  - `details`: dict (optional, zusätzliche Quelle)
    - `issues` / `incorrect_claims` / `claims_incorrect`: Liste von Issue-Objekten
    - Issue-Objekte können `span`, `evidence_quote`, `claim`, `issue_type`, `severity` enthalten

- **coherence**: `AgentResult` (oder dict) mit:
  - `issue_spans`: Liste von `IssueSpan`-Objekten (kanonisch)
  - `details.issues`: Liste (optional, Fallback)

- **readability**: `AgentResult` (oder dict) mit:
  - `issue_spans`: Liste von `IssueSpan`-Objekten (kanonisch)
  - `details.issues`: Liste (optional, Fallback)

**Zusätzlich:**
- `summary_text`: str (der zu analysierende Text, für Span-Extraktion benötigt)

---

## Output Contract

### ExplainabilityResult Schema

```python
class ExplainabilityResult:
    summary: List[str]  # 3-6 Sätze, Executive Summary
    findings: List[Finding]  # Alle Findings, gerankt (höchste Priorität zuerst)
    by_dimension: Dict[Dimension, List[Finding]]  # Findings gruppiert nach Dimension
    top_spans: List[TopSpan]  # Top-K wichtigste Spans (default: K=5)
    stats: ExplainabilityStats  # Statistiken
    version: str  # "m9_v1"
```

### Finding Schema

```python
class Finding:
    id: str  # Deterministische ID (SHA1-basiert)
    dimension: Dimension  # "factuality" | "coherence" | "readability"
    severity: Severity  # "low" | "medium" | "high"
    message: str  # Beschreibung des Problems
    span: Optional[Span]  # Textstelle (start_char, end_char, text)
    evidence: List[EvidenceItem]  # Evidence-Quellen (quotes, claims, raw data)
    recommendation: Optional[str]  # Handlungsempfehlung
    source: Dict[str, Any]  # Provenance (agent, source_list, issue_type, item_index, merged_from, cluster_size)
```

### ExplainabilityStats Schema

```python
class ExplainabilityStats:
    num_findings: int  # Gesamtanzahl Findings
    num_high_severity: int
    num_medium_severity: int
    num_low_severity: int
    coverage_chars: int  # Anzahl Zeichen, die von Spans abgedeckt werden (Union)
    coverage_ratio: float  # coverage_chars / len(summary_text)
```

### Required Fields

- **Jede Dimension muss vorhanden sein** in `by_dimension` (auch wenn leer → leere Liste)
- **Alle Findings müssen** `id`, `dimension`, `severity`, `message` haben
- **Stats müssen** alle Felder haben (auch wenn 0)
- **Top-Spans** müssen `span`, `dimension`, `severity`, `finding_id`, `rank_score` haben

---

## Aggregationsregeln

### 1. Normalisierung

- **Factuality:**
  - Primär: `issue_spans` (kanonisch)
  - Sekundär: `details.issues` / `details.incorrect_claims` (falls vorhanden)
  - Severity-Mapping: `NUMBER`/`DATE` → `high`, `ENTITY`/`NAME`/`LOCATION`/`ORGANIZATION` → `medium`, sonst `medium` (oder raw severity falls String)

- **Coherence/Readability:**
  - Primär: `issue_spans`
  - Fallback: `details.issues`
  - Severity-Mapping: String ("low"/"medium"/"high") oder numerisch (≥0.75 → high, ≥0.4 → medium, sonst low)

### 2. Deduplizierung & Clustern

- **Harte Dedupe:** Findings mit identischer ID werden zusammengeführt (Evidence wird gemerged)
- **Clustern:** Findings mit überlappenden Spans (innerhalb derselben Dimension) werden zu einem Finding zusammengeführt:
  - Primary Finding = höchste Severity (bei Gleichstand: niedrigste ID)
  - Union-Span = min(start) bis max(end)
  - Evidence wird gemerged (dedupliziert über Hash)
  - `source.cluster_size` und `source.cluster_members` werden gesetzt

### 3. Ranking

**Formel:** `score = severity_weight × dimension_weight × (1 + log(span_length))`

- `severity_weight`: low=1.0, medium=2.0, high=3.0
- `dimension_weight`: factuality=1.2, coherence=1.0, readability=0.8
- `span_length`: max(1, end_char - start_char) (falls kein Span: 1)

**Sortierung:** Nach Score (absteigend), bei Gleichstand nach ID (aufsteigend)

### 4. Top-Spans Auswahl

- Top-K (default: K=5) wichtigste Spans
- Keine Duplikate (gleiche (start_char, end_char, dimension) werden übersprungen)
- Sortiert nach `rank_score` (absteigend)

### 5. Stats-Berechnung

- `num_findings`: Anzahl aller Findings (nach Dedupe/Clustern)
- `num_*_severity`: Counts pro Severity-Level
- `coverage_chars`: Union-Länge aller Spans (überlappende Spans werden zusammengeführt)
- `coverage_ratio`: `coverage_chars / len(summary_text)`

### 6. Executive Summary

- Regelbasiert generiert (keine LLM-Calls)
- Enthält:
  - Anzahl Findings (high/medium/low)
  - Schwerpunkt-Dimension (meiste Findings)
  - Top 3 kritische Textstellen (gekürzt auf 70 Zeichen)
  - Prioritätsempfehlung
- Maximal 6 Sätze

---

## Determinismusregeln

### 1. Stabile Sortierung

- **Findings:** Nach Ranking-Score (absteigend), bei Gleichstand nach ID (aufsteigend)
- **by_dimension:** Findings innerhalb jeder Dimension sind stabil sortiert (nach Ranking)
- **Top-Spans:** Nach `rank_score` (absteigend)

### 2. Deterministische IDs

- Finding-ID = SHA1-Hash von: `dimension + severity + issue_type + start_char + end_char + message`
- Format: `f_{hash[:12]}`

### 3. Keine Randomness

- Keine zufälligen Elemente in der Verarbeitung
- Clustern verwendet deterministische Sortierung (nach start_char, end_char)
- Evidence-Merge verwendet deterministischen Hash

### 4. Reproduzierbarkeit

- **Gleicher Input → exakt gleicher Output** (deep equality)
- IDs sind stabil (gleiche Inputs → gleiche IDs)
- Sortierung ist stabil (keine Set-Ordering-Probleme)

---

## Persistence Contract (wenn enabled)

### Postgres

**Tabelle:** `explainability_reports`

- **Required Fields:**
  - `run_id`: int (FK zu `runs.id`)
  - `version`: str (z.B. "m9_v1")
  - `report_json`: JSONB (vollständiges `ExplainabilityResult` als JSON)
  - `created_at`: TIMESTAMPTZ

- **Constraints:**
  - `UNIQUE (run_id, version)` (ein Report pro Run-Version)

### Neo4j (optional)

**Nodes:**
- `Run` (id: run_id)
- `Example` (id: example_id, optional)
- `Explainability` (run_id: run_id, version: str)
- `Finding` (id: finding_id, dimension, severity, message)
- `Span` (start_char, end_char, text)
- `Evidence` (kind, source, quote, data)

**Relationships:**
- `Run` -[:HAS_EXPLAINABILITY]-> `Explainability`
- `Explainability` -[:HAS_FINDING]-> `Finding`
- `Finding` -[:HAS_SPAN]-> `Span`
- `Finding` -[:HAS_EVIDENCE]-> `Evidence`
- `Example` -[:HAS_EXPLAINABILITY]-> `Explainability` (optional)

**Integrity:**
- Keine dangling Nodes (Findings ohne Explainability, Spans ohne Finding)
- `run_id` muss in beiden Stores konsistent sein

---

## Test-Anforderungen

### Contract Tests

- Output hat alle required fields
- Typen stimmen (Dimension enum, Severity literal, etc.)
- Ranges stimmen (score in [0,1], coverage_ratio in [0,1], etc.)
- Jede Dimension ist vorhanden (auch wenn leer)

### Aggregation Tests

- Counts/Stats stimmen
- Top-Spans Auswahl korrekt (höchste rank_score)
- Dedupe/Clustern funktioniert (überlappende Spans werden gemerged)

### Traceability Tests

- Jede Finding referenziert mindestens einen Input-Span (über `source.source_list` und `source.item_index`)
- Evidence IDs referenzieren existierende Evidence-Objekte

### Determinism Tests

- 10x identischer Input → exakt gleicher Output (deep equality)
- Listen sind stabil sortiert (keine random ordering)

### Persistence Tests (optional)

- Save/Query funktioniert (Postgres/Neo4j)
- Keine dangling Nodes
- Cleanup funktioniert (Test-Daten werden entfernt)

---

## Edge Cases

### Leere Inputs

- Keine `issue_spans` → leere Findings-Liste
- Keine Findings → Executive Summary: "Es wurden keine Findings erzeugt..."
- Stats: alle Counts = 0, coverage = 0

### Ungültige Spans

- `start_char` > `end_char` → wird getauscht
- `start_char` < 0 → wird auf 0 gesetzt
- `end_char` > `len(summary_text)` → wird auf `len(summary_text)` gesetzt
- Kein Span → `span = None`, aber Finding existiert trotzdem

### Missing Fields

- Fehlende `severity` → default "medium"
- Fehlende `message` → default "Problem in {dimension} erkannt."
- Fehlende `issue_type` → None (wird in Severity-Mapping berücksichtigt)

---

## Versionierung

- **Version:** `m9_v1` (fest im Code)
- **Änderungen:** Bei Breaking Changes → neue Version (z.B. `m9_v2`)
- **Vergleichbarkeit:** Reports mit gleicher Version sind vergleichbar

