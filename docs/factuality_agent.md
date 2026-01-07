# Factuality Agent

## Übersicht

Der Factuality-Agent prüft automatisch, ob Behauptungen (Claims) in einer Zusammenfassung durch den Originalartikel belegt sind.

### Pipeline

1. **Satz-Segmentierung**: Summary wird in Sätze zerlegt (mit stabilen Char-Spans)
2. **Claim-Extraktion**: Pro Satz werden atomare, faktische Claims extrahiert (LLM-basiert)
3. **Claim-Verifikation**: Jeder Claim wird gegen den Artikel verifiziert (Evidence-Gate)
4. **Aggregation**: Claims werden zu Satz-Labels aggregiert → Gesamtscore
5. **Issue-Span-Generierung**: Problematische Stellen werden als IssueSpans markiert (repräsentativ pro Satz)

## Evidence-Gate

### Kernidee

Das Evidence-Gate reduziert False Positives erheblich, indem es eine harte Regel durchsetzt:

**"incorrect" wird nur vergeben, wenn belastbare Evidence (wörtliches Zitat aus dem Artikel) gefunden wurde.**

### Harte Regel

- `label in {"correct", "incorrect"}` **nur** wenn `evidence_found == True`
- `evidence_found == True` **nur** wenn:
  - `selected_evidence_index >= 0`
  - `evidence_quote` ist nicht leer
  - `evidence_quote` ist wörtlicher Substring der ausgewählten Passage (nach Normalisierung)

### Invarianten

1. `selected_evidence_index == -1` → `evidence_quote = None` → `evidence_found = False`
2. `selected_evidence_index >= 0` → `evidence_quote` muss nicht-leer sein UND in Passage vorkommen → `evidence_found = True`
3. Coverage-Fail führt **nicht** zu Label-Downgrade, sondern nur zu Confidence-Clamp
4. Gate-Logik sitzt ausschließlich im `ClaimVerifier` (keine zusätzlichen Safety-Downgrades im Agent)
5. Verdict (incorrect/uncertain) ist unabhängig von Severity (low/medium/high)

### Optional: Evidence für "correct"

Wenn `require_evidence_for_correct=True`:
- "correct" wird nur vergeben, wenn `evidence_found == True`
- Ohne Evidence → "uncertain"

## Artefakte

### Evaluation-Artefakte

- `results/evaluation/evidence_gate_test/results_*.json`:
  - Metriken (TP/TN/FP/FN, Recall, Precision, F1, Balanced Accuracy)
  - Coverage/Abstention Statistiken
  - Evidence-Stats auf Claim-Level und IssueSpan-Level
  - Ground Truth Distribution

- `results/evaluation/evidence_gate_test/debug_claims.jsonl`:
  - Pro Claim: verdict, evidence_found, confidence
  - Gate-Entscheidungsweg (label_raw → label_final, gate_reason)
  - Evidence-Selection-Details (selection_reason, evidence_quote)
  - Coverage-Status (coverage_ok, coverage_note)
  - Raw LLM-Output (für Debugging)

- `results/evaluation/evidence_gate_test/error_cases.jsonl`:
  - Nur False Positives und False Negatives
  - Top Issue Span, Top Claims, Claim-Label-Counts
  - Für gezielte Fehleranalyse

## Komponenten

### FactualityAgent

Haupt-Agent, orchestriert die gesamte Pipeline:
- Satz-Segmentierung
- Claim-Extraktion (via `ClaimExtractor`)
- Claim-Verifikation (via `ClaimVerifier`)
- Score-Berechnung
- Issue-Span-Generierung

### ClaimExtractor

Extrahiert atomare, faktische Claims aus einem Satz (LLM-basiert).

### ClaimVerifier

Verifiziert einen Claim gegen den Artikel:
- Evidence-Retrieval (via `EvidenceRetriever`)
- LLM-basierte Verifikation
- Evidence-Validierung (Invarianten)
- Gate-Logik (correct/incorrect nur mit Evidence)

### EvidenceRetriever

Deterministischer Retriever für Evidence-Passagen:
- Sliding-Window-Passagen (2-3 Sätze pro Passage)
- Jaccard-Similarity Scoring
- Zahl/Entity-Boost

## Debug-Felder

Der Factuality-Agent verwendet `__dict__`-basierte Debug-Felder für Traceability und Analyse:

### Warum existieren diese Felder?

- **Gate-Entscheidungsweg nachvollziehen**: `label_raw`, `label_final`, `gate_reason` zeigen, wie ein Claim durch die Gate-Logik verarbeitet wurde
- **Evidence-Selection analysieren**: `evidence_selection_reason`, `coverage_ok`, `coverage_note` zeigen, warum Evidence gefunden/nicht gefunden wurde
- **LLM-Output debuggen**: `raw_verifier_output`, `selected_evidence_index_raw`, `evidence_quote_raw` zeigen, was das LLM ursprünglich ausgab

### Verwendung

- **Debug-Export**: Diese Felder werden in `debug_claims.jsonl` und `error_cases.jsonl` exportiert
- **Analyse-Scripts**: `scripts/test_evidence_gate_eval.py` und `scripts/print_factuality_eval_bullets.py` nutzen diese Felder für Statistiken
- **Nicht für produktive Logik**: Die produktive Gate-Logik und Metrikberechnung hängen nicht von diesen Feldern ab

### Technische Umsetzung

Die Debug-Felder werden über `claim.__dict__` gesetzt (in `claim_verifier.py`) und über `c.__dict__.get()` gelesen (in `factuality_agent.py`). Dies ermöglicht:
- Dynamische Felder ohne Claim-Model-Änderungen
- Rückwärtskompatibilität (alte Runs ohne Debug-Felder funktionieren weiterhin)
- Saubere Trennung zwischen produktiven Feldern (im Claim-Model) und Debug-Feldern (in `__dict__`)

## Stabilitätsentscheidung

Für M10 (Bachelorarbeit) werden folgende Komponenten **eingefroren** und nicht mehr geändert:

- **Evidence-Gate**: Invarianten, Gate-Regeln, Coverage-Checks bleiben unverändert
- **Score-Berechnung**: Satz-basierte Aggregation bleibt unverändert
- **Output-Formate**: `results_*.json`, `debug_claims.jsonl`, `error_cases.jsonl` Struktur bleibt unverändert
- **IssueSpan-Generierung**: Repräsentative Span-Logik bleibt unverändert

**Optional (außerhalb M10-Scope):**
- Retriever-Verbesserungen (z.B. bessere Scoring-Heuristiken)
- Quote-Matching-Robustheit (z.B. erweiterte Normalisierung)
- Recall-Optimierungen (würden Trade-offs mit Precision erfordern)

