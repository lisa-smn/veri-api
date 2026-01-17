# 2. Grundlagen

## 2.2 Evaluationsansätze: klassische Metriken vs LLM-as-a-Judge vs Agentensysteme

Die Evaluation von automatisch generierten Summaries stellt eine zentrale Herausforderung dar. Traditionelle Ansätze basieren auf einfachen Formeln, während neuere Methoden Large Language Models (LLMs) als Evaluatoren einsetzen. Ein dritter Ansatz nutzt strukturierte Agentensysteme, die sowohl Evaluierung als auch Erklärbarkeit bieten.

### Klassische Metriken: Readability-Formeln

Klassische Readability-Formeln wie Flesch Reading Ease, Flesch-Kincaid Grade Level und Gunning Fog Index basieren auf oberflächlichen Textmerkmalen wie Satzlänge, Silbenanzahl und Wortkomplexität. Diese Metriken sind schnell berechenbar, kostengünstig und vollständig reproduzierbar, da sie deterministisch auf rein statistischen Eigenschaften des Textes basieren.

**Empirische Ergebnisse zeigen jedoch eine sehr schwache Korrelation mit menschlichen Bewertungen.** In unserer Evaluation auf dem SummEval-Datensatz (n=200) erreichten klassische Baselines einen Spearman-Korrelationskoeffizienten von ρ ≈ -0.05, was praktisch keine Korrelation bedeutet. Dies liegt daran, dass Readability-Formeln keine semantische Analyse durchführen: Sie können nicht erkennen, ob ein Text inhaltlich kohärent ist, ob Referenzen klar sind oder ob die Struktur logisch ist. Ein Text mit kurzen Sätzen und einfachen Wörtern kann dennoch inhaltlich verwirrend oder schlecht strukturiert sein.

**Vorteile klassischer Metriken:**
- Schnelle Berechnung (Millisekunden)
- Keine API-Kosten
- Vollständig reproduzierbar
- Keine Abhängigkeit von externen Services

**Nachteile:**
- Keine semantische Analyse
- Schlechte Korrelation mit Human Ratings
- Keine Erklärbarkeit (nur ein Score)

### LLM-as-a-Judge: Semantische Evaluation durch LLMs

Der Ansatz "LLM-as-a-Judge" nutzt Large Language Models (z.B. GPT-4o-mini) als Evaluatoren, die Summaries direkt bewerten. Der LLM erhält eine Zusammenfassung und eine Referenz (Artikel oder Human Rating) und gibt einen Score oder eine Bewertung aus. Dieser Ansatz profitiert vom semantischen Verständnis moderner LLMs: Sie können Kohärenz, Lesbarkeit und sogar faktische Korrektheit beurteilen, ohne auf einfache statistische Merkmale beschränkt zu sein.

**In unserer Evaluation erreichte der LLM-as-a-Judge für Readability einen Spearman-Korrelationskoeffizienten von ρ ≈ 0.280 (n=200), was eine moderate Korrelation darstellt.** Dies ist deutlich besser als klassische Formeln, zeigt aber auch, dass LLMs nicht perfekt mit menschlichen Bewertungen übereinstimmen.

**Stärken von LLM-as-a-Judge:**
- Semantisches Verständnis
- Flexible Bewertung (verschiedene Dimensionen möglich)
- Bessere Korrelation mit Human Ratings als klassische Metriken

**Schwächen:**
- **Bias:** LLMs können systematische Vorurteile haben (z.B. Präferenz für bestimmte Schreibstile)
- **Kosten:** Jede Evaluation erfordert einen API-Call (bei n=200: signifikante Kosten)
- **Reproduzierbarkeit:** Abhängig von Temperature, Prompting, Model-Version
- **Latenz:** Langsamer als klassische Formeln (Sekunden statt Millisekunden)
- **Fehlende Struktur:** Output ist oft nur ein Score ohne detaillierte Erklärung

### Agentensysteme: Strukturierte, erklärbare Evaluation

Agentensysteme kombinieren die semantische Analyse von LLMs mit einer strukturierten Pipeline, die spezifische Aspekte der Summary analysiert. In unserem System arbeiten drei Agenten parallel:

1. **Factuality Agent:** Extrahiert Claims, sucht Evidence im Artikel, verifiziert Claims gegen Evidence
2. **Coherence Agent:** Analysiert Satzstruktur, logische Übergänge, Referenzklarheit
3. **Readability Agent:** Bewertet Satzkomplexität, Struktur, Lesbarkeit

**Jeder Agent liefert nicht nur einen Score, sondern auch strukturierte IssueSpans:** Positionen im Text, Issue-Typen (z.B. NUMBER, DATE, CONTRADICTION) und Verdicts (correct/incorrect/uncertain). Diese strukturierten Outputs werden dann durch ein Explainability-Modul aggregiert, das Findings mit Severity-Levels (low, medium, high) und Top-Spans generiert.

**In unserer Evaluation erreichte der Readability Agent einen Spearman-Korrelationskoeffizienten von ρ ≈ 0.402 (n=200), was die beste Korrelation mit Human Ratings darstellt.** Dies liegt deutlich über dem LLM-as-a-Judge (ρ ≈ 0.280) und weit über klassischen Baselines (ρ ≈ -0.05).

**Vorteile von Agentensystemen:**
- **Strukturierte Analyse:** Spezifische Issue-Typen, nicht nur ein Score
- **Erklärbarkeit:** Jedes Finding ist nachvollziehbar (Spans, Evidence, Severity)
- **Persistenz:** Ergebnisse werden in Postgres + Neo4j gespeichert (Traceability)
- **Bessere Korrelation:** Übertrifft sowohl klassische Metriken als auch LLM-as-a-Judge
- **Reproduzierbarkeit:** Deterministische Aggregation, Seed-basierte Runs

**Nachteile:**
- **Komplexität:** Mehrschichtige Pipeline (Extraction → Verification → Aggregation)
- **Latenz:** Langsamer als klassische Formeln (aber vergleichbar mit LLM-as-a-Judge)
- **Kosten:** Verwendet LLMs für Extraction und Verification (aber strukturiert, nicht nur Scoring)

### Vergleich und Auswahl

Die Wahl des Evaluationsansatzes hängt von den Anforderungen ab:

- **Schnelle, kostengünstige Evaluation ohne Erklärbarkeit:** Klassische Metriken (aber schlechte Korrelation)
- **Moderate Korrelation, einfache Integration:** LLM-as-a-Judge (aber Bias, Kosten, Reproduzierbarkeit)
- **Beste Korrelation + Erklärbarkeit + Persistenz:** Agentensysteme (aber komplexer)

**Für unsere Anwendung (Verifikationssystem mit Explainability-Anforderungen) ist der Agentenansatz optimal:** Er bietet die beste Korrelation mit Human Ratings, strukturierte Erklärungen und vollständige Traceability durch Persistenz. Die zusätzliche Komplexität ist gerechtfertigt, da sie nicht nur bessere Scores liefert, sondern auch nachvollziehbare Findings, die für Nutzerinnen und Nutzer verständlich sind.

---

**Empirische Basis:**
- Readability Agent: `results/evaluation/readability/readability_20260116_170832_gpt-4o-mini_v1_seed42/`
- Readability Judge: Spearman ρ ≈ 0.280 (aus gleichem Run)
- Readability Baselines: `results/evaluation/baselines/baselines_readability_flesch_fk_fog_20260116_175246_seed42/`
- Status-Dokumentation: `docs/status/readability_status.md`

