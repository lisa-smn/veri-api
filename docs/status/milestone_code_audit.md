# Milestone Code Audit

**Datum:** 2026-01-17  
**Zweck:** Vergleich der Milestone-Dokumente mit dem aktuellen Code-Stand und Korrektur von Diskrepanzen

---

## M1 – Projekt-Setup & Minimal-API

### Diff-Report

**✅ Korrekt:**
- Projektstruktur (`app/`, `api/`, `core/`, `services/`, `models/`, `db/`)
- FastAPI als Hauptframework
- Endpunkte `/health` und `/verify` existieren (vgl. `app/api/routes.py:14-20, 20-37`)
- Konfiguration über `.env` und `pydantic-settings` (vgl. `app/core/config.py:1-26`)
- Abhängigkeiten (FastAPI, Uvicorn, Pydantic, SQLAlchemy, Neo4j-Treiber)

**❌ Falsch/Veraltet:**
- **`main.py` im Projekt-Root:** Existiert nicht. Tatsächlicher Einstiegspunkt ist `app/server.py` (vgl. `app/server.py:1-63`). Server wird über `uvicorn app.server:app` gestartet, nicht `uvicorn main:app`.

**⚠️ Teilweise korrekt:**
- `/verify` Endpunkt: Doku sagt "fest kodierte Scores", aber aktuell führt er echte Pipeline aus (vgl. `app/api/routes.py:20-37`). Das ist korrekt für den aktuellen Stand, aber M1 beschreibt den initialen Dummy-Zustand.

**❓ Nicht auffindbar:**
- Keine

---

## M2 – Datenmodell & Persistenzschicht (Postgres)

### Diff-Report

**✅ Korrekt:**
- Postgres 16 als Docker-Container
- Schema mit Tabellen: `datasets`, `articles`, `summaries`, `runs`, `verification_results`, `errors` (als `run_errors`), `explanations` (vgl. `app/db/postgres/schema.sql:37-118`)
- Enum-Typen: `summary_source`, `run_type`, `run_status`, `verification_dimension` (vgl. `app/db/postgres/schema.sql:2-34`)
- SQLAlchemy-Session-Management (vgl. `app/db/postgres/session.py`)
- `get_db()`-Dependency für API-Endpunkte (vgl. `app/db/postgres/session.py`)
- `DATABASE_URL` über `pydantic-settings` (vgl. `app/core/config.py:17`)

**⚠️ Teilweise korrekt:**
- **Tabelle `errors`:** Im Schema heißt sie `run_errors` (vgl. `app/db/postgres/schema.sql:99-106`), nicht `errors`. M2-Doku verwendet den falschen Namen.
- **`explanations` Tabelle:** Wird noch verwendet (vgl. `app/db/postgres/persistence.py:169-190`), aber ist optional (nur wenn `agent.explanation` vorhanden ist).

**❓ Nicht auffindbar:**
- `scripts/m2_insert_test_data.py`: Nicht im Repo gefunden. Möglicherweise entfernt oder umbenannt.

---

## M3 – Evaluations-Kern (Pipeline & Agenten)

### Diff-Report

**✅ Korrekt:**
- `VerificationPipeline` mit Agent-Orchestrierung (vgl. `app/pipeline/verification_pipeline.py:29-61`)
- Dummy-Agenten wurden später durch echte Agenten ersetzt (M5, M7, M8)
- Pydantic-Modelle (`AgentResult`, `PipelineResult`) (vgl. `app/models/pydantic.py:33-91`)
- Integration in `/verify` Endpunkt (vgl. `app/api/routes.py:20-37`)

**⚠️ Teilweise korrekt:**
- **Persistenz:** M3 sagt "Funktionen implementiert zum Speichern von Artikeln, Summaries, Runs, Agenten-Ergebnissen, Erklärungen". Tatsächlich wurden diese Funktionen in M2 implementiert (`store_article_and_summary`, `store_verification_run` in `app/db/postgres/persistence.py:33-271`). M3 nutzt diese Funktionen, implementiert sie aber nicht.

**❓ Nicht auffindbar:**
- Keine

---

## M4 – Graph-Erweiterung (Neo4j-Integration)

### Diff-Report

**✅ Korrekt:**
- Neo4j-Integration als Docker-Container
- Graph-Persistenz über `write_verification_graph()` (vgl. `app/db/neo4j/graph_persistence.py:12-49`)
- Best-effort Schreiblogik (vgl. `app/db/postgres/persistence.py:253-268`)
- `NEO4J_ENABLED` Environment-Variable (vgl. `app/db/neo4j/graph_persistence.py:23`)

**❌ Falsch/Veraltet:**
- **Neo4j-Client:** M4 sagt "`neo4j_client.py`", tatsächlich heißt die Datei `neo4j_client.py` im Ordner `app/db/neo4j/` (vgl. `app/db/neo4j/neo4j_client.py`). ✅ Korrekt, nur Pfad präzisieren.
- **Konfiguration:** M4 sagt "Konfiguration über `.env`: `NEO4J_URL`, `NEO4J_USER`, `NEO4J_PASSWORD`". Tatsächlich verwendet der Code `settings.neo4j_url`, `settings.neo4j_user`, `settings.neo4j_password` aus `app/core/config.py` (vgl. `app/core/config.py:19-21`, `app/db/neo4j/neo4j_client.py:12-13`). Die ENV-Vars werden über `pydantic-settings` geladen, nicht direkt.
- **Graph-Modell Node `Error`:** M4 sagt "`Error`-Knoten". Tatsächlich verwendet der Code Dual-Label `IssueSpan:Error` (vgl. `app/db/neo4j/graph_persistence.py:113`). Der Knoten heißt `IssueSpan`, hat aber auch das Label `Error` für Abwärtskompatibilität.
- **Relationship `HAS_ERROR`:** M4 sagt "`HAS_ERROR`". Tatsächlich verwendet der Code `HAS_ISSUE_SPAN` (vgl. `app/db/neo4j/graph_persistence.py:124`). `HAS_ERROR` existiert nicht.

**⚠️ Teilweise korrekt:**
- **Graph-Modell:** M4 listet Nodes und Relationships auf, aber die tatsächliche Struktur ist komplexer:
  - Nodes: `Article`, `Summary`, `Run`, `Metric`, `IssueSpan:Error` (Dual-Label)
  - Relationships: `HAS_SUMMARY` (Article → Summary), `EVALUATES` (Run → Summary), `HAS_METRIC` (Summary → Metric), `HAS_ISSUE_SPAN` (Metric → IssueSpan)
  - M4 erwähnt `HAS_ERROR`, was nicht existiert.

**❓ Nicht auffindbar:**
- Keine

---

## M5 – Factuality-Agent & Verifikationslogik

### Diff-Report

**✅ Korrekt:**
- `LLMClient` Interface definiert (vgl. `app/llm/llm_client.py:5-9`)
- `OpenAIClient` und `FakeLLMClient` Implementierungen (vgl. `app/llm/openai_client.py:11-25`, `app/llm/fake_client.py:6-26`)
- `TEST_MODE` entscheidet zwischen echter und Fake-LLM (vgl. `app/pipeline/verification_pipeline.py:26, 31-34`)
- Factuality-Agent mit Satz-für-Satz-Analyse (vgl. `app/services/agents/factuality/factuality_agent.py:99-150`)
- JSON-Parsing, Score-Berechnung, Issue-Extraktion
- Integration in Pipeline (vgl. `app/pipeline/verification_pipeline.py:36`)

**⚠️ Teilweise korrekt:**
- **`AgentResult.errors`:** M5 sagt "`errors`". Tatsächlich verwendet der Code `issue_spans` als kanonisches Feld, akzeptiert aber `errors` als Legacy-Alias (vgl. `app/models/pydantic.py:43-49`). Serialisierung erfolgt immer als `issue_spans`.

**❓ Nicht auffindbar:**
- Keine

---

## M6 – Claim-basierter Factuality-Agent & Evaluationsinfrastruktur

### Diff-Report

**✅ Korrekt:**
- Claim-Extraction (`LLMClaimExtractor`) (vgl. `app/services/agents/factuality/claim_extractor.py:24-328`)
- Evidence-Retrieval (deterministisch: Sliding-Window + Jaccard-Similarity) (vgl. `app/services/agents/factuality/evidence_retriever.py:45-98`)
- Claim-Verification (`LLMClaimVerifier`) (vgl. `app/services/agents/factuality/claim_verifier.py:35-382`)
- Evidence-Gate-Logik (vgl. `app/services/agents/factuality/claim_verifier.py:319-369`)
- Evaluationsinfrastruktur für FRANK/FineSumFact

**❌ Falsch/Veraltet:**
- **Evaluationsskript `scripts/eval_frank.py`:** Existiert nicht. Tatsächliche Skripte sind:
  - `scripts/eval_frank_factuality_agent_on_manifest.py` (Agent-Evaluation)
  - `scripts/eval_frank_factuality_baselines.py` (Baseline-Evaluation)
  - `scripts/eval_frank_factuality_llm_judge.py` (Judge-Evaluation)
- **FineSumFact-Konverter:** M6 sagt "`scripts/convert_finesumfact.py`". Existiert (vgl. `scripts/convert_finesumfact.py`). ✅ Korrekt.

**⚠️ Teilweise korrekt:**
- **FRANK-Dataset:** M6 sagt "Dataset bereinigt (`frank_clean.jsonl`)". Muss im Repo verifiziert werden, ob diese Datei existiert.

**❓ Nicht auffindbar:**
- Keine

---

## M7 – Kohärenz-Agent

### Diff-Report

**✅ Korrekt:**
- `CoherenceAgent` implementiert (vgl. `app/services/agents/coherence/coherence_agent.py:15-150`)
- `LLMCoherenceEvaluator` (vgl. `app/services/agents/coherence/coherence_verifier.py:117-160`)
- Issue-Typen: LOGICAL_INCONSISTENCY, CONTRADICTION, REDUNDANCY, ORDERING, OTHER (vgl. `app/services/agents/coherence/coherence_verifier.py:140`)
- Score-Berechnung (0.0-1.0)
- Integration in Pipeline (vgl. `app/pipeline/verification_pipeline.py:37`)
- SummEval-Evaluation

**⚠️ Teilweise korrekt:**
- **Issue-Typen:** M7 erwähnt "CONTRADICTION, MISSING_TRANSITION, UNCLEAR_REFERENCE". Tatsächlich verwendet der Code: LOGICAL_INCONSISTENCY, CONTRADICTION, REDUNDANCY, ORDERING, OTHER (vgl. `app/services/agents/coherence/coherence_verifier.py:140`). "MISSING_TRANSITION" und "UNCLEAR_REFERENCE" existieren nicht als explizite Typen (könnten unter "ORDERING" oder "OTHER" fallen).

**❓ Nicht auffindbar:**
- Keine

---

## M8 – Readability-Agent

### Diff-Report

**✅ Korrekt:**
- `ReadabilityAgent` implementiert (vgl. `app/services/agents/readability/readability_agent.py:27-284`)
- `LLMReadabilityEvaluator` (vgl. `app/services/agents/readability/readability_verifier.py`)
- Issue-Typen: LONG_SENTENCE, COMPLEX_NESTING, PUNCTUATION_OVERLOAD, HARD_TO_PARSE (vgl. `app/services/agents/readability/readability_verifier.py:255`)
- Integration in Pipeline (vgl. `app/pipeline/verification_pipeline.py:38`)
- Postgres/Neo4j Persistenz

**❌ Falsch/Veraltet:**
- **Readability v2:** M8 erwähnt v2 nicht. Tatsächlich unterstützt der Agent zwei Prompt-Versionen:
  - v1: Score 0.0-1.0 (vgl. `app/services/agents/readability/readability_verifier.py:159-206`)
  - v2: Rubrik-basiertes 1-5 Rating (`score_raw_1_to_5`), wird intern zu 0-1 normalisiert (vgl. `app/services/agents/readability/readability_verifier.py:208-254, 81-86`)
  - Constraint: Wenn `score_raw_1_to_5 <= 2`, dann min 1 issue (vgl. `app/services/agents/readability/readability_verifier.py:269`)
  - Raw score wird zusätzlich gespeichert (vgl. `app/services/agents/readability/readability_verifier.py:352-356`)

**⚠️ Teilweise korrekt:**
- **Issue-Typen:** M8 sagt "COMPLEX_SENTENCE, POOR_STRUCTURE, UNCLEAR_REFERENCE". Tatsächlich verwendet der Code: LONG_SENTENCE, COMPLEX_NESTING, PUNCTUATION_OVERLOAD, HARD_TO_PARSE (vgl. `app/services/agents/readability/readability_verifier.py:255`). "COMPLEX_SENTENCE" vs "LONG_SENTENCE" ist eine Diskrepanz, "POOR_STRUCTURE" und "UNCLEAR_REFERENCE" existieren nicht.

**❓ Nicht auffindbar:**
- Keine

---

## M9 – Explainability-Modul

### Diff-Report

**✅ Korrekt:**
- `ExplainabilityService` implementiert (vgl. `app/services/explainability/explainability_service.py:274-311`)
- `ExplainabilityResult` Pydantic-Modell (vgl. `app/services/explainability/explainability_models.py:81-87`)
- Pipeline-Integration (vgl. `app/pipeline/verification_pipeline.py:40, 59`)
- API-Integration (vgl. `app/api/routes.py:31`)
- Postgres Persistenz (`explainability_reports` Tabelle) (vgl. `app/db/postgres/schema.sql:120-129`, `app/db/postgres/persistence.py:218-240`)
- Versionierung (`m9_v1`) (vgl. `app/services/explainability/explainability_service.py:275`)

**❌ Falsch/Veraltet:**
- **`ErrorSpan`:** M9 sagt "`ErrorSpan` wurde erweitert um `issue_type`". Tatsächlich heißt das Modell `IssueSpan` (vgl. `app/models/pydantic.py:8-30`), nicht `ErrorSpan`. Der Name wurde geändert, um zu reflektieren, dass nicht alle Issues Fehler sind (uncertain Issues sind keine klaren Fehler).

**⚠️ Teilweise korrekt:**
- **Datenquelle:** M9 sagt "Primäre Datenquelle für Explainability ist `AgentResult.issue_spans`". Korrekt, aber das Feld heißt `issue_spans`, nicht `errors` (vgl. `app/models/pydantic.py:46-49`).

**❓ Nicht auffindbar:**
- Keine

---

## M10 – Evaluation & Vergleich mit klassischen Metriken

### Diff-Report

**✅ Korrekt:**
- Evaluationsskripte (`scripts/run_m10_factuality.py`, `scripts/aggregate_factuality_runs.py`)
- Run-Management mit Caching
- `run_tag` vs `prompt_version` Mapping (vgl. `scripts/run_m10_factuality.py:200-228`)
- Baselines (Flesch, Flesch-Kincaid, Gunning Fog)
- LLM-as-a-Judge
- Metriken (Spearman ρ, Pearson r, MAE, F1, AUROC)

**⚠️ Teilweise korrekt:**
- **Run-Config-Format:** M10 sagt "Einheitliches Run-Config-Format (JSON/YAML) pro Experiment". Tatsächlich verwendet das System YAML-Configs (vgl. `configs/m10_factuality_runs.yaml`), nicht JSON.

**❓ Nicht auffindbar:**
- Keine

---

## M11 – Orchestrierung & Integration

### Diff-Report

**✅ Korrekt:**
- Status: Entfällt (nicht im Thesis-Scope) (vgl. `docs/milestones/M11_orchestrierung_und_integration.md:4-8`)

**❓ Nicht auffindbar:**
- Keine

---

## M12 – Streamlit Dashboard (UI)

### Diff-Report

**✅ Korrekt:**
- Streamlit-UI implementiert (`ui/app.py`)
- Drei Tabs: Verify, Runs, Status
- Dataset-Auswahl (FRANK, SummEval)
- Explainability-Visualisierung
- Error-Injection für Demo-Zwecke

**❓ Nicht auffindbar:**
- Keine kritischen Diskrepanzen gefunden. M12-Doku ist aktuell und korrekt.

---

## Zusammenfassung der kritischen Korrekturen

1. **M1:** `main.py` → `app/server.py`
2. **M2:** `errors` Tabelle → `run_errors` Tabelle
3. **M4:** `Error` Node → `IssueSpan:Error` (Dual-Label), `HAS_ERROR` → `HAS_ISSUE_SPAN`, Neo4j Config über `settings.*` nicht direkt ENV-Vars
4. **M5:** `errors` Feld → `issue_spans` (mit Legacy-Alias)
5. **M6:** `scripts/eval_frank.py` → `scripts/eval_frank_factuality_agent_on_manifest.py` (und andere)
6. **M7:** Issue-Typen präzisieren (MISSING_TRANSITION, UNCLEAR_REFERENCE existieren nicht explizit)
7. **M8:** Readability v2 fehlt komplett (Normalisierung, Constraints, Raw-Score-Speicherung)
8. **M9:** `ErrorSpan` → `IssueSpan`

---

## Durchgeführte Korrekturen

Die folgenden Milestone-Dokumente wurden korrigiert:

1. **M1:** `main.py` → `app/server.py`, Startbefehl korrigiert
2. **M2:** `errors` → `run_errors` Tabelle, Code-Referenzen hinzugefügt
3. **M3:** Persistenz-Funktionen korrekt referenziert (aus M2, nicht M3)
4. **M4:** Graphschema korrigiert (`IssueSpan:Error` Dual-Label, `HAS_ISSUE_SPAN` statt `HAS_ERROR`), Neo4j Config über `settings.*`, Code-Referenzen hinzugefügt
5. **M5:** `errors` → `issue_spans` (mit Legacy-Alias), Code-Referenz hinzugefügt
6. **M6:** Evaluationsskripte korrigiert (mehrere Skripte statt einem), Code-Referenzen hinzugefügt
7. **M7:** Issue-Typen präzisiert (LOGICAL_INCONSISTENCY, CONTRADICTION, REDUNDANCY, ORDERING, OTHER), Code-Referenzen hinzugefügt
8. **M8:** Readability v2 vollständig dokumentiert (Normalisierung, Constraints, Raw-Score-Speicherung), Issue-Typen korrigiert, Code-Referenzen hinzugefügt
9. **M9:** `ErrorSpan` → `IssueSpan`, Code-Referenzen hinzugefügt
10. **M10:** Run-Config-Format präzisiert (YAML statt JSON/YAML), Code-Referenz hinzugefügt

Alle Korrekturen wurden direkt in die Milestone-Dateien eingearbeitet. Code-Referenzen (Datei:Zeilen) wurden ergänzt, wo möglich.

