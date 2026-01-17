# Thesis: Theory Outline

**Datum:** 2026-01-17  
**Status:** Draft

---

## Kapitelstruktur

### 1. Problem & Motivation

- **Risiken von automatisch generierten Summaries:**
  - Faktische Fehler (Factuality)
  - Inkohärenz (Coherence)
  - Schlechte Lesbarkeit (Readability)
  - Fehlende Transparenz (Explainability)

- **Need for Verification:**
  - Automatisierte Qualitätskontrolle
  - Erklärbare Ergebnisse
  - Reproduzierbare Evaluation

---

### 2. Grundlagen

#### 2.1 LLM Summarization & typische Fehler

- **LLM-basierte Summarization:**
  - Transformer-Architekturen (GPT, etc.)
  - Halluzinationen
  - Inkonsistenzen
  - Strukturelle Probleme

- **Typische Fehlerkategorien:**
  - **Factuality:** Falsche Zahlen, Daten, Entitäten
  - **Coherence:** Logische Brüche, Widersprüche, fehlende Übergänge
  - **Readability:** Zu lange Sätze, schlechte Struktur, unklare Referenzen

#### 2.2 Evaluationsansätze: klassische Metriken vs LLM-as-a-Judge vs Agentensysteme

**Klassische Metriken:**
- Readability-Formeln (Flesch, Flesch-Kincaid, Gunning Fog)
- Korrelation mit Human Ratings: **schwach** (Spearman ρ ≈ -0.05)
- Vorteile: Schnell, kostengünstig, reproduzierbar
- Nachteile: Keine semantische Analyse, schlechte Korrelation

**LLM-as-a-Judge:**
- Verwendung von LLMs als Evaluatoren
- Stärken: Semantisches Verständnis, flexible Bewertung
- Schwächen: Bias, Kosten, Reproduzierbarkeit (Temperature, Prompting)
- Korrelation mit Human Ratings: **moderat** (Spearman ρ ≈ 0.28-0.40)

**Agentensysteme:**
- Strukturierte Analyse (Claims, Evidence, Issues)
- Erklärbare Ergebnisse (IssueSpans, Findings)
- Persistenz (Postgres + Neo4j für Traceability)
- Korrelation mit Human Ratings: **gut** (Spearman ρ ≈ 0.40)
- Vorteile: Strukturiert, erklärbar, nachvollziehbar
- Nachteile: Komplexer, langsamer als Formeln

#### 2.3 Explainability: Begriff, Anforderungen, Trade-offs

- **Begriff:**
  - Erklärbarkeit vs. Interpretierbarkeit
  - Post-hoc vs. Intrinsische Explainability

- **Anforderungen:**
  - **Traceability:** Jedes Finding referenziert originale Agent-Outputs
  - **Severity:** Klassifikation (low, medium, high)
  - **Spans:** Exakte Text-Positionen
  - **Evidence:** Quellenangaben (für Factuality)

- **Trade-offs:**
  - Granularität vs. Übersichtlichkeit
  - Vollständigkeit vs. Performance
  - Automatisch vs. Manuell

---

### 3. Methodik / Systemdesign

#### 3.1 Pipeline-Architektur

- **Verification Pipeline:**
  - Input: Article + Summary
  - Agents: Factuality, Coherence, Readability
  - Aggregation: Scores (0-1)
  - Explainability: Findings + Spans
  - Persistence: Postgres + Neo4j

#### 3.2 Agenten-Design

**Factuality Agent:**
- Claim Extraction (LLM)
- Evidence Retrieval (Sliding Window)
- Claim Verification (LLM + Evidence-Gate)
- Output: IssueSpans + Score

**Coherence Agent:**
- Sentence-level Analysis
- Structure Scoring
- Output: Score

**Readability Agent:**
- Sentence Complexity Analysis
- Normalization (1-5 → 0-1)
- Output: Score

#### 3.3 Datenmodell

- **Postgres:**
  - Runs, Articles, Summaries, Explainability Reports
  - JSONB für flexible Metadaten

- **Neo4j:**
  - Graph: Run → Example → Explainability → Finding → Span
  - Cross-Store Consistency: `run_id`

#### 3.4 Persistenz

- **Postgres:** Relationale Daten (Runs, Results)
- **Neo4j:** Graph-Daten (Traceability, Relationships)
- **Cross-Store Contract:** Einheitliche `run_id`

---

### 4. Evaluation Setup

#### 4.1 Datasets

- **FRANK:** Factuality (Binary: `has_error` true/false)
- **SummEval:** Coherence + Readability (Kontinuierlich: 1-5, normalisiert zu 0-1)

#### 4.2 Metriken

**Factuality:**
- Precision, Recall, F1, Balanced Accuracy, MCC, AUROC

**Coherence/Readability:**
- Pearson r, Spearman ρ, MAE, RMSE, R² (vs. Human Ratings)

#### 4.3 Baselines

- **Classical:** Flesch, Flesch-Kincaid, Gunning Fog
- **LLM-as-a-Judge:** GPT-4o-mini (optional, `ENABLE_LLM_JUDGE=true`)

#### 4.4 Reproducibility & CI

- **Git Tags:** `readability-final-2026-01-16`, `thesis-snapshot-2026-01-16`
- **CI:** pytest, ruff, sanity checks
- **Environment Variables:** Dokumentiert

---

### 5. Ergebnisse & Interpretation

#### 5.1 Readability

- **Agent:** Spearman ρ ≈ 0.402 (vs. Human Ratings)
- **Judge:** Spearman ρ ≈ 0.280
- **Baselines:** Spearman ρ ≈ -0.05 (keine Korrelation)
- **Interpretation:** Agent übertrifft Baselines deutlich, Judge moderat

#### 5.2 Factuality

- **Judge Baseline:** Precision, Recall, F1, Balanced Accuracy
- **Interpretation:** LLM-as-a-Judge als Baseline etabliert

#### 5.3 Coherence

- **Agent:** Korrelation mit Human Ratings
- **Interpretation:** Strukturierte Analyse funktioniert

---

### 6. Limitationen & Future Work

#### 6.1 Limitationen

- **UI:** Demo/Prototype, nicht production-ready
- **Judge:** Optional, langsam, kostenintensiv
- **Persistence:** Optional, DB-Setup erforderlich
- **Datasets:** SummEval ohne Referenzen → keine ROUGE/BERTScore

#### 6.2 Future Work

- **Multi-Agent Orchestration:** Parallelisierung
- **Calibration:** Post-hoc Score-Kalibrierung
- **Explainability LLM-Calls:** Erweiterte Erklärungen
- **Production-Ready UI:** Auth, Rate-Limiting
- **Additional Baselines:** Mehr klassische Metriken

---

### 7. Fazit

- **Zusammenfassung:** Agentensystem übertrifft klassische Baselines
- **Beitrag:** Strukturierte, erklärbare Verifikation
- **Ausblick:** Production-Ready System, erweiterte Baselines

---

## Literatur (Placeholder)

- [Wird ergänzt]

