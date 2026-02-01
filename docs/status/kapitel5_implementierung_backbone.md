# Kapitel 5 "Implementierung" – Backbone

**Zweck:** Strukturierte Grundlage für 3-seitiges Implementierungskapitel der Bachelorarbeit  
**Datum:** 2026-01-17  
**Quellen:** Code-Repository, Milestone-Dokumentation (M1–M12), Status-Reports

---

## A) Quellenliste (Dokumentation)

| Datei | Kurzinhalt | Relevanz für Kapitel 5 |
|-------|------------|------------------------|
| `docs/milestones/README.md` | Übersicht M1–M12, System-Architektur (Final) | ⭐⭐⭐ Hoch: Chronologie, Komponenten-Übersicht |
| `docs/milestones/M1_setup.md` | Projekt-Setup, FastAPI-Grundstruktur | ⭐⭐ Mittel: API-Entry-Point, Konfiguration |
| `docs/milestones/M2_datenmodell.md` | Postgres-Schema, SQLAlchemy-Integration | ⭐⭐⭐ Hoch: Persistenz-Implementierung |
| `docs/milestones/M6_claim_basierter_factuality_agent_und_evaluationsinfrastruktur.md` | Claim-Extraction, Evidence-Retrieval, Claim-Verification | ⭐⭐⭐ Hoch: Factuality-Agent-Details |
| `docs/milestones/M7_kohärenz_agent.md` | Coherence-Agent, Satzstruktur-Analyse | ⭐⭐ Mittel: Coherence-Implementierung |
| `docs/milestones/M8_readability_agent.md` | Readability-Agent, Satzkomplexität | ⭐⭐ Mittel: Readability-Implementierung |
| `docs/milestones/M9_explainability_modul.md` | Explainability-Service, Findings-Aggregation | ⭐⭐⭐ Hoch: Explainability-Transformation |
| `docs/milestones/M10_evaluation_setup.md` | Evaluationsskripte, Run-Configs, Metriken | ⭐⭐ Mittel: Reproduzierbarkeit, Evaluation-Setup |
| `docs/status/architektur_komplett.md` | Vollständige Architektur-Dokumentation | ⭐⭐⭐ Hoch: Komponenten-Übersicht, Datenmodelle |
| `docs/status/factuality_status.md` | Factuality-Evaluation-Ergebnisse | ⭐ Niedrig: Nur Zahlen, keine Implementierung |
| `docs/status/explainability_spec.md` | Explainability-Contract, Aggregationsregeln | ⭐⭐⭐ Hoch: Explainability-Details |
| `configs/m10_factuality_runs.yaml` | YAML-Run-Konfigurationen für Evaluation | ⭐⭐ Mittel: Reproduzierbarkeit, Config-Format |
| `app/db/postgres/schema.sql` | PostgreSQL-Schema-Definition | ⭐⭐⭐ Hoch: Datenmodell-Implementierung |
| `tests/unit/test_evidence_gate_refactored.py` | Evidence-Gate Unit-Tests | ⭐⭐ Mittel: Test-Nachweise für Gate-Logik |

---

## B) Komponentenkarte

| Komponente | Kurzbeschreibung | Nachweis: Pfade |
|------------|------------------|-----------------|
| **API/Entry** | FastAPI-App mit `/verify` Endpunkt, Dependency Injection für DB | `app/server.py:20-31`, `app/api/routes.py:20-22` |
| **Pipeline/Orchestrierung** | `VerificationPipeline` orchestriert drei Agenten, berechnet Overall-Score, integriert Explainability | `app/pipeline/verification_pipeline.py:29-61` |
| **Factuality Agent** | Claim-Extraction → Evidence-Retrieval → Claim-Verification → Score-Aggregation, Evidence-Gate-Logik | `app/services/agents/factuality/factuality_agent.py:24-633`, `app/services/agents/factuality/claim_verifier.py` |
| **Coherence Agent** | LLM-basierte Bewertung von logischer Konsistenz, Satzstruktur, Referenzklarheit | `app/services/agents/coherence/coherence_agent.py:15-150` |
| **Readability Agent** | LLM-basierte Bewertung von Lesbarkeit, Satzkomplexität, Struktur | `app/services/agents/readability/readability_agent.py:27-284` |
| **Explainability Service** | Transformation AgentResult → ExplainabilityResult: Normalisierung, Deduplizierung, Clustering, Ranking, Executive Summary | `app/services/explainability/explainability_service.py:281-311` |
| **PostgreSQL Persistenz** | SQLAlchemy-basierte Speicherung von Runs, Verification Results, Explanations | `app/db/postgres/persistence.py`, `app/db/postgres/schema.sql:37-160` |
| **Neo4j Persistenz** | Graph-basierte Speicherung (optional): Article, Summary, Metric, Error Nodes | `app/db/neo4j/graph_persistence.py:write_verification_graph()` |
| **LLM-Client** | Abstraktion für LLM-Aufrufe (OpenAI-Implementierung, FakeLLM für Tests) | `app/llm/openai_client.py`, `app/llm/fake_client.py` |
| **Konfiguration** | Pydantic-Settings für ENV-Variablen (DATABASE_URL, NEO4J_URL, OPENAI_API_KEY) | `app/core/config.py:4-26` |
| **Evaluationsskripte** | Reproduzierbare Evaluation mit YAML-Configs, Bootstrap-Konfidenzintervallen | `scripts/run_m10_factuality.py`, `scripts/eval_factuality_binary_v2.py`, `configs/m10_factuality_runs.yaml` |
| **Tests** | Unit-Tests für Evidence-Gate, Agent-Contracts, Explainability-Determinismus | `tests/unit/test_evidence_gate_refactored.py`, `tests/explainability/test_explainability_determinism.py` |

---

## C) Milestone-Mapping

| Milestone | Ziel | Implementierte Bausteine | Wichtigste Dateien | Wichtigste Tests | Ergebnisartefakte |
|-----------|------|--------------------------|-------------------|------------------|-------------------|
| **M1** | Projekt-Setup & Minimal-API | FastAPI-Grundstruktur, Konfiguration über `.env` | `app/server.py`, `app/api/routes.py`, `app/core/config.py` | - | - |
| **M2** | Datenmodell & Persistenzschicht (Postgres) | Postgres-Schema, SQLAlchemy-Integration | `app/db/postgres/schema.sql`, `app/db/postgres/session.py` | - | - |
| **M3** | Eval-Core-Skelett | Pipeline-Grundgerüst, Dummy-Agenten | `app/pipeline/verification_pipeline.py` | - | - |
| **M4** | Graph-Modell (Neo4j) | Neo4j-Integration, Graph-Persistenz | `app/db/neo4j/graph_persistence.py` | - | - |
| **M5** | Factuality-Agent & Verifikationslogik | Erster echter Agent, LLM-Abstraktion | `app/services/agents/factuality/factuality_agent.py` | - | - |
| **M6** | Claim-basierter Factuality-Agent & Evaluationsinfrastruktur | Claim-Extraction, Evidence-Retrieval, Claim-Verification, Evaluationsskripte | `app/services/agents/factuality/claim_extractor.py`, `app/services/agents/factuality/claim_verifier.py`, `app/services/agents/factuality/evidence_retriever.py`, `scripts/eval_factuality_binary_v2.py` | `tests/unit/test_factuality_agent_unit.py` | - |
| **M7** | Kohärenz-Agent | Coherence-Agent, Satzstruktur-Analyse | `app/services/agents/coherence/coherence_agent.py` | `tests/unit/test_coherence_agent.py` | - |
| **M8** | Readability-Agent | Readability-Agent, Satzkomplexität | `app/services/agents/readability/readability_agent.py` | `tests/unit/test_readability_agent.py` | - |
| **M9** | Explainability-Modul | Findings-Aggregation, Severity-Levels, Top-Spans, Deduplizierung | `app/services/explainability/explainability_service.py` | `tests/explainability/test_explainability_determinism.py`, `tests/explainability/test_explainability_contract.py` | - |
| **M10** | Evaluation & Vergleich mit klassischen Metriken | Evaluationsskripte, YAML-Run-Configs, Bootstrap-Konfidenzintervalle, LLM-as-a-Judge | `scripts/run_m10_factuality.py`, `configs/m10_factuality_runs.yaml`, `app/services/judges/llm_judge.py` | - | `results/evaluation/factuality/`, `results/evaluation/readability/`, `results/evaluation/baselines/` |
| **M12** | Streamlit Dashboard (UI) | Interaktives Dashboard für Demo und Inspektion | `ui/app.py` | `tests/ui/test_dataset_loader.py` | - |

---

## D) Kapitel-Backbone für 3 Seiten

### 5.1 Technische Umsetzung (Überblick)

**Stichpunkte:**
- **Architektur:** Modulares System mit FastAPI als API-Layer, drei spezialisierten Agenten (Factuality, Coherence, Readability), Explainability-Service zur Aggregation, duale Persistenz (PostgreSQL + Neo4j)
- **Entwicklung:** Chronologische Entwicklung über 12 Meilensteine (M1–M12), beginnend mit Setup (M1) und Datenmodell (M2), über Agent-Implementierungen (M5–M8), bis hin zu Explainability (M9) und Evaluation (M10)
- **API-Entry:** `app/server.py` als FastAPI-App mit `/verify` Endpunkt, Dependency Injection für DB-Sessions, Lifespan-Management für Neo4j
- **Pipeline:** `VerificationPipeline` orchestriert sequenzielle Ausführung der drei Agenten, berechnet Overall-Score als arithmetisches Mittel, integriert Explainability-Service deterministisch
- **Agenten:** Jeder Agent implementiert `run(article, summary, meta)` → `AgentResult` mit Score (0–1), `issue_spans`, `explanation`, `details`
- **Explainability:** Transformiert Agent-Outputs in einheitliches `ExplainabilityResult` mit Findings, Top-Spans, Executive Summary (Version `m9_v1`)
- **Persistenz:** PostgreSQL für relationale Speicherung (Runs, Verification Results, Explanations), Neo4j optional für Graph-Repräsentation (Article, Summary, Metric, Error Nodes)
- **Reproduzierbarkeit:** YAML-Run-Configs (`configs/m10_factuality_runs.yaml`) für Evaluation, Seed-basierte LLM-Calls (seed=42), Bootstrap-Konfidenzintervalle für Metriken
- **Tests:** Unit-Tests für Evidence-Gate-Logik, Agent-Contracts, Explainability-Determinismus, Integration-Tests für Pipeline

**Belegstellen:**
- `docs/milestones/README.md:183-194` (System-Architektur Final)
- `app/server.py:20-31` (API-Entry)
- `app/pipeline/verification_pipeline.py:29-61` (Pipeline-Orchestrierung)
- `docs/status/architektur_komplett.md` (Komponenten-Übersicht)

---

### 5.2 Pipelineablauf

**Ablauf in 8 Schritten:**

1. **API-Request:** `POST /verify` mit `VerifyRequest` (article_text, summary_text, meta) → `app/api/routes.py:20-22`
2. **Service-Orchestrierung:** `VerificationService.verify()` speichert Article/Summary in PostgreSQL, erstellt Run-Eintrag → `app/services/verification_service.py`
3. **Pipeline-Initialisierung:** `VerificationPipeline` initialisiert drei Agenten (Factuality, Coherence, Readability) und Explainability-Service → `app/pipeline/verification_pipeline.py:29-40`
4. **Agent-Ausführung (sequenziell):**
   - Factuality: Claim-Extraction → Evidence-Retrieval → Claim-Verification → Score-Aggregation → `AgentResult`
   - Coherence: LLM-basierte Bewertung → Issue-Extraktion → Score-Berechnung → `AgentResult`
   - Readability: LLM-basierte Bewertung → Issue-Extraktion → Score-Berechnung → `AgentResult`
5. **Overall-Score:** Arithmetisches Mittel der drei Agent-Scores → `app/pipeline/verification_pipeline.py:47`
6. **Explainability-Transformation:** `ExplainabilityService.build()` normalisiert, dedupliziert, clustert, rankt Findings, generiert Executive Summary → `app/services/explainability/explainability_service.py:281-311`
7. **Persistenz:** Speicherung in PostgreSQL (Runs, Verification Results, Explanations) und optional Neo4j (Graph) → `app/db/postgres/persistence.py:store_verification_run()`
8. **Response:** `VerifyResponse` mit run_id, Scores, issue_spans, explainability → `app/api/routes.py:21-22`

**Optional ASCII-Diagramm:**
```
POST /verify
  ↓
VerificationService (Article/Summary → DB)
  ↓
VerificationPipeline.run()
  ├─ FactualityAgent.run() → AgentResult
  ├─ CoherenceAgent.run() → AgentResult
  ├─ ReadabilityAgent.run() → AgentResult
  └─ Overall Score = (F + C + R) / 3
  ↓
ExplainabilityService.build() → ExplainabilityResult
  ↓
Persistence (PostgreSQL + Neo4j)
  ↓
VerifyResponse
```

**Belegstellen:**
- `app/pipeline/verification_pipeline.py:42-61` (Pipeline-Ablauf)
- `app/services/explainability/explainability_service.py:281-311` (Explainability-Transformation)
- `app/db/postgres/persistence.py:store_verification_run()` (Persistenz)

---

### 5.3 Beispiel Factuality-Agent (detailliert)

**Stichpunkte:**
- **Claim-Extraction:** `LLMClaimExtractor` extrahiert atomare, überprüfbare Claims aus Summary-Sätzen via LLM-Prompt → `app/services/agents/factuality/claim_extractor.py`
- **Evidence-Retrieval:** `EvidenceRetriever` sucht relevante Passagen im Artikel via Sliding-Window + Jaccard-Similarity + Boosting für Zahlen/Entities → `app/services/agents/factuality/evidence_retriever.py`
- **Claim-Verification:** `LLMClaimVerifier` verifiziert Claims gegen Evidence, liefert Label (correct/incorrect/uncertain), Confidence, Issue-Type (NUMBER/DATE/ENTITY/OTHER) → `app/services/agents/factuality/claim_verifier.py`
- **Evidence-Gate:** Kernlogik: "incorrect" nur erlaubt, wenn `evidence_found == True`; "incorrect" ohne Evidence → "uncertain" (Confidence clamp auf 0.5); "correct" ohne Evidence (wenn `require_evidence_for_correct=True`) → "uncertain" (Confidence clamp auf 0.55) → `app/services/agents/factuality/claim_verifier.py:_apply_gate()`
- **Score-Aggregation:** Satzweise Labels → Summary-Score: `correct=1.0`, `incorrect=0.0`, `uncertain=0.5`, gewichtet nach Confidence → `app/services/agents/factuality/factuality_agent.py:run()`
- **IssueSpans:** Jeder Claim generiert `IssueSpan` mit `start_char`, `end_char`, `message`, `severity`, `issue_type`, `verdict` (incorrect/uncertain) → `app/services/agents/factuality/factuality_agent.py:_build_issue_spans()`

**Belegstellen:**
- `app/services/agents/factuality/factuality_agent.py:99-633` (Agent-Implementierung)
- `app/services/agents/factuality/claim_verifier.py:_apply_gate()` (Evidence-Gate-Logik)
- `tests/unit/test_evidence_gate_refactored.py` (Gate-Tests)

---

### 5.4 Coherence/Readability (knapp)

**Coherence Agent:**
- LLM-basierte Bewertung von logischer Konsistenz, Satzstruktur, Referenzklarheit → `app/services/agents/coherence/coherence_agent.py:69-150`
- Issue-Typen: CONTRADICTION, MISSING_TRANSITION, UNCLEAR_REFERENCE
- Score-Berechnung: LLM-Output (0–1) direkt als Score
- Integration: Vollständig in Pipeline integriert, IssueSpans für problematische Textstellen

**Readability Agent:**
- LLM-basierte Bewertung von Lesbarkeit, Satzkomplexität, Struktur → `app/services/agents/readability/readability_agent.py:82-284`
- Issue-Typen: COMPLEX_SENTENCE, POOR_STRUCTURE, UNCLEAR_REFERENCE
- Score-Berechnung: LLM-Output (0–1) direkt als Score
- Integration: Vollständig in Pipeline integriert, IssueSpans für problematische Textstellen

**Belegstellen:**
- `app/services/agents/coherence/coherence_agent.py:15-150` (Coherence)
- `app/services/agents/readability/readability_agent.py:27-284` (Readability)

---

### 5.5 Beispiel Explainability-Ausgabe (Struktur)

**Stichpunkte:**
- **Input:** `PipelineResult` mit drei `AgentResult`-Objekten (Factuality, Coherence, Readability), `summary_text`
- **Transformation:** 6-Schritte-Pipeline:
  1. Normalisierung: Agent-Outputs → Findings (Dimension, Severity, Span, Message)
  2. Deduplizierung: Überlappende Findings (>50% Char-Overlap) werden zusammengeführt
  3. Clustering: Findings mit ähnlichen Spans werden gruppiert
  4. Ranking: `rank_score = severity_weight × dimension_weight × log(span_length)`
  5. Top-Spans: Top-K Findings nach Rank-Score (default: K=5)
  6. Executive Summary: Regelbasierte Zusammenfassung aus Findings
- **Output:** `ExplainabilityResult` mit:
  - `version`: "m9_v1" (versioniert für Reproduzierbarkeit)
  - `summary`: Liste von 3–6 Sätzen (Executive Summary)
  - `findings`: Liste von Findings, gerankt (höchste Priorität zuerst)
  - `by_dimension`: Findings gruppiert nach Dimension (factuality, coherence, readability)
  - `top_spans`: Top-K wichtigste Spans mit `span`, `dimension`, `severity`, `finding_id`, `rank_score`
  - `stats`: Basisstatistiken (num_findings, num_high/medium/low_severity, coverage_chars, coverage_ratio)

**Mini JSON-Snippet (< 20 Zeilen):**
```json
{
  "version": "m9_v1",
  "summary": [
    "Die Zusammenfassung enthält 3 faktische Probleme (NUMBER, DATE) und 1 Kohärenz-Problem.",
    "Die wichtigsten Probleme betreffen Zahlenangaben und Datumsangaben."
  ],
  "findings": [
    {
      "id": "sha1_hash",
      "dimension": "factuality",
      "severity": "high",
      "message": "Falsche Zahlenangabe: '2020' statt '2021'",
      "span": {"start_char": 45, "end_char": 50, "text": "2020"},
      "recommendation": "Zahlenangaben mit dem Artikel abgleichen."
    }
  ],
  "top_spans": [...],
  "stats": {
    "num_findings": 4,
    "num_high_severity": 2,
    "coverage_ratio": 0.15
  }
}
```

**Belegstellen:**
- `app/services/explainability/explainability_service.py:281-311` (Build-Methode)
- `docs/status/explainability_spec.md` (Output-Contract)
- `docs/milestones/M9_explainability_modul.md` (Designentscheidungen)

---

## E) Fehlende Infos/ToDos

**NICHT GEFUNDEN / UNSICHER:**
- **Prompt-Versionen:** Konkrete Prompt-Texte für Claim-Extraction, Claim-Verification, Coherence, Readability sind nicht direkt im Code sichtbar (vermutlich in `app/services/agents/*/prompts.py` oder inline). **Suchspur:** `grep -r "prompt" app/services/agents/` → gefunden, aber nicht vollständig gelesen.
- **LLM-Model-Details:** Welche genauen Model-Parameter (Temperature, Max-Tokens) werden verwendet? **Suchspur:** `app/llm/openai_client.py` → vermutlich Defaults, nicht explizit dokumentiert.
- **Neo4j-Schema-Details:** Vollständige Liste aller Node-Labels und Relation-Types ist nicht explizit dokumentiert. **Suchspur:** `app/db/neo4j/graph_persistence.py` → Code zeigt: Article, Summary, Metric, Error Nodes, aber keine vollständige Dokumentation.
- **Ablation-Modi:** Details zu `ablation_mode` (no_claims, sentence_only, no_spans) sind im Code vorhanden, aber nicht in Milestone-Dokumentation erklärt. **Suchspur:** `app/services/agents/factuality/ablation_extractor.py`, `ablation_verifier.py` → gefunden, aber nicht vollständig analysiert.

**Empfehlung:** Diese Punkte können im Kapitel als "Designentscheidungen" oder "Konfigurationsparameter" kurz erwähnt werden, ohne detaillierte Code-Referenzen.

