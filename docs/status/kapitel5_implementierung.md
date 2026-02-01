# Kapitel 5: Implementierung

Dieses Kapitel beschreibt die technische Umsetzung des agentenbasierten Verifikationssystems. Die Implementierung erfolgte schrittweise über zehn Meilensteine (M1–M10, M12), die von der Grundstruktur bis zur vollständigen Evaluationsinfrastruktur reichen (vgl. `docs/milestones/README.md`). Das System ist als modulare FastAPI-Anwendung realisiert, die drei spezialisierte Agenten orchestriert und deren Ergebnisse zu erklärbaren Reports aggregiert.

## 5.1 Implementierungsüberblick

Das System besteht aus fünf zentralen Komponenten: einer API-Schicht, einer Verifikationspipeline, drei Agenten (Factuality, Coherence, Readability), einem Explainability-Modul und einer dualen Persistenzschicht (PostgreSQL und Neo4j). Die Architektur folgt einem klaren Schichtenmodell, das Trennung von Verantwortlichkeiten und Testbarkeit gewährleistet (vgl. `docs/status/architektur_komplett.md`).

**API-Layer:** Der Einstiegspunkt ist eine FastAPI-Anwendung (`app/server.py:51-52`), die beim Startup kritische Umgebungsvariablen validiert (`app/server.py:16-40`). Der Hauptendpunkt `/verify` (`app/api/routes.py:20-37`) nimmt `VerifyRequest`-Objekte entgegen und gibt strukturierte `VerifyResponse`-Objekte zurück, die Scores, IssueSpans und Explainability-Reports enthalten.

**Pipeline:** Die `VerificationPipeline` (`app/pipeline/verification_pipeline.py:29-61`) orchestriert die sequenzielle Ausführung der drei Agenten, berechnet einen Overall-Score als arithmetisches Mittel und ruft anschließend den Explainability-Service auf. Die Pipeline ist vollständig unabhängig von der API-Schicht und kann auch direkt als Bibliothek verwendet werden.

**Agenten:** Jeder Agent implementiert das einheitliche `AgentResult`-Interface (`app/models/pydantic.py:33-54`) und liefert einen normalisierten Score (0.0–1.0), strukturierte IssueSpans und optionale Details. Der Factuality-Agent ist claim-basiert implementiert (M6, vgl. `docs/milestones/M6_claim_basierter_factuality_agent_und_evaluationsinfrastruktur.md`), während Coherence und Readability direkt LLM-basierte Bewertungen durchführen (M7, M8).

**Explainability:** Das Explainability-Modul (`app/services/explainability/explainability_service.py:76-690`) transformiert Agent-Outputs in einen deterministischen, versionierten Report (Version `m9_v1`, vgl. `docs/status/explainability_spec.md`). Die Transformation umfasst Normalisierung, Deduplizierung, Ranking und die Generierung eines Executive Summary.

**Persistenz:** Ergebnisse werden sowohl in PostgreSQL (relational) als auch optional in Neo4j (graphbasiert) gespeichert. Das PostgreSQL-Schema (`app/db/postgres/schema.sql`) umfasst Tabellen für Runs, Verification-Results, Explanations und Explainability-Reports. Neo4j speichert Runs als Graph mit Nodes für Article, Summary, Metric und Error (`app/db/neo4j/graph_persistence.py:51-152`).

Die Entwicklung erfolgte inkrementell: M1 etablierte die Grundstruktur (vgl. `docs/milestones/M1_setup.md`), M2 das Datenmodell (`docs/milestones/M2_datenmodell.md`), M6 den claim-basierten Factuality-Agent, M7–M8 die weiteren Agenten, M9 das Explainability-Modul (`docs/milestones/M9_explainability_modul.md`) und M10 die Evaluationsinfrastruktur (`docs/milestones/M10_evaluation_setup.md`).

## 5.2 Verifikationspipeline und Datenfluss

Der Datenfluss von einem API-Request bis zur Response umfasst acht Schritte (vgl. `docs/status/architektur_komplett.md`, Abschnitt E):

1. **HTTP-Request:** Der Client sendet `POST /verify` mit `VerifyRequest` (`app/api/routes.py:20-37`).
2. **Service-Orchestrierung:** `VerificationService.verify()` (`app/services/verification_service.py:49-171`) empfängt Request und DB-Session.
3. **Persistenz (Artikel/Summary):** `store_article_and_summary()` speichert Artikel und Summary in PostgreSQL (`app/db/postgres/persistence.py:33-96`).
4. **Pipeline-Ausführung:** `VerificationPipeline.run()` (`app/pipeline/verification_pipeline.py:42-61`) führt die drei Agenten sequenziell aus.
5. **Agent-Ausführung:** Jeder Agent (`FactualityAgent.run()`, `CoherenceAgent.run()`, `ReadabilityAgent.run()`) liefert ein `AgentResult` mit Score, IssueSpans und Details.
6. **Overall-Score:** Arithmetisches Mittel der drei Agent-Scores (`app/pipeline/verification_pipeline.py:47`).
7. **Explainability:** `ExplainabilityService.build()` transformiert Agent-Outputs in einen strukturierten Report (`app/pipeline/verification_pipeline.py:59`).
8. **Persistenz (Run + Ergebnisse):** `store_verification_run()` speichert Run, Verification-Results, Explanations und Explainability-Report in PostgreSQL (`app/db/postgres/persistence.py:102-271`).

Die Pipeline ist deterministisch: Bei identischen Inputs (Artikel, Summary, LLM-Parameter) ergeben sich identische Agent-Outputs, und die Explainability-Transformation ist vollständig regelbasiert ohne LLM-Calls (vgl. `docs/status/explainability_spec.md`, Abschnitt "Determinismusregeln").

## 5.3 Factuality-Agent als Referenzimplementierung

Der Factuality-Agent dient als Referenzimplementierung für die claim-basierte Verifikation (M6). Der Agent durchläuft vier Schritte: Claim-Extraction, Evidence-Retrieval, Claim-Verification mit Evidence-Gate und Score-Aggregation.

**Claim-Extraction:** Der `LLMClaimExtractor` (`app/services/agents/factuality/claim_extractor.py`) zerlegt die Summary in atomare, überprüfbare Claims. Jeder Claim enthält Text, Satzindex und optionale Metadaten. Falls die Extraktion fehlschlägt, wird der gesamte Satz als Fallback-Claim verwendet (`app/services/agents/factuality/factuality_agent.py:99-323`).

**Evidence-Retrieval:** Der `EvidenceRetriever` (`app/services/agents/factuality/evidence_retriever.py:45-171`) sucht relevante Passagen im Artikel mittels Sliding-Window-Ansatz und Jaccard-Similarity-Scoring. Zahlen und Entitäten werden geboostet, um präzise Evidence zu finden.

**Claim-Verification:** Der `LLMClaimVerifier` (`app/services/agents/factuality/claim_verifier.py:75`) verifiziert jeden Claim gegen die gefundene Evidence. Das LLM gibt ein Label (correct/incorrect/uncertain), eine Confidence und optional eine Evidence-Quote zurück.

**Evidence-Gate:** Die zentrale Logik des Evidence-Gates ist in `_apply_gate()` implementiert (`app/services/agents/factuality/claim_verifier.py:200-250`). Das Gate erlaubt "incorrect"-Labels nur, wenn `evidence_found == True` ist. Fehlt Evidence für ein "incorrect"-Label, wird es zu "uncertain" downgraded und die Confidence auf maximal 0.5 geklemmt. Diese Regel reduziert False Positives erheblich, da Behauptungen nur als falsch markiert werden, wenn belastbare Evidence (wörtliche Zitate) im Artikel gefunden wurde. Die Gate-Logik ist durch Unit-Tests abgesichert (`tests/unit/test_evidence_gate_refactored.py`).

**Score-Aggregation:** Der Agent aggregiert Claim-Labels satzweise: `correct = 1.0`, `incorrect = 0.0`, `uncertain = 0.5` (neutral gewichtet, vgl. `app/services/agents/factuality/factuality_agent.py:327-633`). Der Gesamtscore ist der Durchschnitt über alle Sätze.

**IssueSpans:** Aus Claims mit `label in ("incorrect", "uncertain")` werden IssueSpans generiert (`app/services/agents/factuality/factuality_agent.py:450-550`). Jeder Span enthält Char-Positionen, Message, Severity, Issue-Type (z.B. NUMBER, DATE, ENTITY) und Verdict.

## 5.4 Explainability: Aggregation und Determinismus

Das Explainability-Modul transformiert die heterogenen Agent-Outputs in einen einheitlichen, deterministischen Report (Version `m9_v1`, vgl. `docs/status/explainability_spec.md`). Die Transformation erfolgt in sechs Schritten:

1. **Normalisierung:** Agent-Outputs werden in ein gemeinsames `Finding`-Format übersetzt. Factuality nutzt spezielle Severity-Mappings (NUMBER/DATE → high, ENTITY → medium), während Coherence/Readability generisch normalisiert werden (`app/services/explainability/explainability_service.py:76-690`).

2. **Deduplizierung:** Findings mit identischer ID (SHA1-Hash aus Dimension, Severity, Issue-Type, Span-Position, Message) werden zusammengeführt.

3. **Clustering:** Findings mit überlappenden Spans (innerhalb derselben Dimension) werden zu einem Finding gemerged. Das Primary Finding ist das mit höchster Severity, der Union-Span deckt alle überlappenden Bereiche ab.

4. **Ranking:** Findings werden nach `rank_score = severity_weight × dimension_weight × (1 + log(span_length))` priorisiert (`app/services/explainability/explainability_service.py:56-63`). Factuality hat höchste Dimension-Gewichtung (1.2), Readability niedrigste (0.8).

5. **Top-Spans:** Die Top-K wichtigsten Spans (default: K=5) werden extrahiert, sortiert nach Rank-Score.

6. **Executive Summary:** Ein regelbasierter Text (3–6 Sätze) wird aus Findings abgeleitet, ohne LLM-Calls. Der Summary enthält Anzahl Findings, Schwerpunkt-Dimension und Top-3 kritische Textstellen.

**Determinismus:** Der Report ist vollständig deterministisch: Gleicher Input führt zu exakt gleichem Output (deep equality). Finding-IDs sind stabil (SHA1-Hash), Sortierungen sind deterministisch, und es gibt keine Randomness in der Verarbeitung. Dies wird durch Determinismus-Tests verifiziert (`tests/explainability/test_explainability_determinism.py`).

## 5.5 Reproduzierbarkeit und Evaluation-Artefakte

Die Evaluationsinfrastruktur (M10) gewährleistet vollständige Reproduzierbarkeit durch YAML-basierte Run-Konfigurationen, deterministische Caches und standardisierte Artefakte.

**Run-Konfigurationen:** Jeder Evaluation-Run wird in einer YAML-Datei definiert (`configs/m10_factuality_runs.yaml`). Die Konfiguration umfasst Dataset-Pfad, LLM-Modell, Temperatur, Seed, `run_tag` (für Cache-Identifikation), Decision-Modi (issues/score/either/both), Thresholds und Ablationsmodi (z.B. `no_claims`, `sentence_only`, `no_spans`).

**Run-Manifeste:** Jeder Run erzeugt ein Manifest (`.json`) in `results/evaluation/runs/results/` (`scripts/run_m10_factuality.py:759-787`). Das Manifest enthält `run_id`, Metriken (Precision, Recall, F1, AUROC), Anzahl Beispiele, Config-Snapshot, Commit-Hash und `created_at` (ISO 8601 UTC). Das Manifest dient als zentrale Referenz für Reproduzierbarkeit.

**Caches:** LLM-Antworten werden deterministisch gecacht (`scripts/run_m10_factuality.py:154-170`). Der Cache-Key basiert auf Artikel-Text, Summary-Text, Modell und `run_tag`. Cache-Dateien (`.jsonl`) werden in `results/evaluation/cache/` gespeichert und ermöglichen schnelle Re-Runs ohne erneute LLM-Calls.

**Run-Dokumentation:** Jeder Run erzeugt eine Markdown-Dokumentation (`results/evaluation/runs/docs/<run_id>.md`) mit Metriken, Konfidenzintervallen, Fehlerprofilen und Reproduzierbarkeits-Informationen (`scripts/run_m10_factuality.py:609-756`).

**Beispiele:** Example-Level-Ergebnisse werden als JSONL gespeichert (`<run_id>_examples.jsonl`), um spätere Analysen zu ermöglichen.

**Aggregation:** Das Skript `scripts/aggregate_factuality_runs.py` aggregiert mehrere Runs zu Vergleichstabellen (CSV/Markdown), die Metriken, `run_tag`, Modell und Dataset-Signaturen enthalten.

**Reproduzierbarkeit:** Jeder Run ist durch Git-Commit-Hash, `run_tag`, Modell-Version und Prompt-Versionen eindeutig identifizierbar. Die Artefakte (Manifest, Docs, Examples) ermöglichen vollständige Nachvollziehbarkeit ohne erneute Ausführung.

## Zusammenfassung

Die Implementierung des agentenbasierten Verifikationssystems erfolgte schrittweise über zehn Meilensteine, von der Grundstruktur bis zur vollständigen Evaluationsinfrastruktur. Das System ist als modulare FastAPI-Anwendung realisiert, die drei spezialisierte Agenten orchestriert und deren Ergebnisse zu erklärbaren Reports aggregiert. Der Factuality-Agent verwendet eine claim-basierte Verifikation mit Evidence-Gate, das False Positives reduziert, indem "incorrect"-Labels nur bei vorhandener Evidence erlaubt werden. Das Explainability-Modul transformiert Agent-Outputs deterministisch in strukturierte Reports (Version `m9_v1`), die Findings, Top-Spans und Executive Summaries enthalten. Die Evaluationsinfrastruktur gewährleistet vollständige Reproduzierbarkeit durch YAML-Konfigurationen, deterministische Caches und standardisierte Artefakte (Run-Manifeste, Dokumentation, Beispiele). Alle Implementierungsentscheidungen sind im Repository dokumentiert und durch Tests abgesichert.

