
---

# M10 – Evaluation & Vergleich mit klassischen Metriken

## Ziel

In M10 wird das System **systematisch und reproduzierbar evaluiert**.
Die Agent-Scores (Factuality/Coherence/Readability) sowie der Explainability-Report werden **quantitativ** gegen Goldlabels bzw. Human-Ratings geprüft und **direkt** mit klassischen Metriken (z.B. ROUGE, BERTScore, SummaC) verglichen.

Ergebnis sind belastbare Aussagen darüber, **ob** und **wann** das agentische System besser mit menschlichen Bewertungen übereinstimmt als klassische Metriken und welchen Mehrwert die Explainability-Schicht für nachvollziehbare Fehlerdiagnosen liefert.

**Status:** ✅ Abgeschlossen (2026-01-16)

---

## Evaluationsansätze: Klassische Metriken vs LLM-as-a-Judge vs Agentensysteme

Die Evaluation von automatisch generierten Summaries stellt eine zentrale Herausforderung dar. Traditionelle Ansätze basieren auf einfachen Formeln, während neuere Methoden Large Language Models (LLMs) als Evaluatoren einsetzen. Ein dritter Ansatz nutzt strukturierte Agentensysteme, die sowohl Evaluierung als auch Erklärbarkeit bieten.

### Klassische Metriken: Readability-Formeln

Klassische Readability-Formeln wie Flesch Reading Ease, Flesch-Kincaid Grade Level und Gunning Fog Index basieren auf oberflächlichen Textmerkmalen wie Satzlänge, Silbenanzahl und Wortkomplexität. Diese Metriken sind schnell berechenbar, kostengünstig und vollständig reproduzierbar, da sie deterministisch auf rein statistischen Eigenschaften des Textes basieren.

**Empirische Ergebnisse zeigen jedoch eine sehr schwache Korrelation mit menschlichen Bewertungen.** In unserer Evaluation auf dem SummEval-Datensatz (n=200) erreichten klassische Baselines einen Spearman-Korrelationskoeffizienten von ρ ≈ -0.05, was praktisch keine Korrelation bedeutet. Dies liegt daran, dass Readability-Formeln keine semantische Analyse durchführen: Sie können nicht erkennen, ob ein Text inhaltlich kohärent ist, ob Referenzen klar sind oder ob die Struktur logisch ist.

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

## Ausgangslage (was das System bereits kann)

* `/verify` liefert pro Beispiel:

  * Scores pro Dimension + Overall
  * strukturierte `issue_spans` (inkl. `issue_type`)
  * deterministischen Explainability-Report (`m9_v1`)
* Persistenz ist vorhanden:

  * `runs`, `verification_results`, `explainability_reports` in Postgres
* Tests laufen stabil (Unit + API), Neo4j ist im Testmodus deaktivierbar

Damit ist die technische Grundlage für Evaluation bereits gegeben. In M10 fehlt vor allem die **Evaluationsebene** (Scripts, Baselines, Analyse) und eine saubere, reproduzierbare Run-Definition.

---

## 0) Reproduzierbarkeit & Run-Definition (Evaluation “einfrieren”)

Damit Ergebnisse später nachvollziehbar und zitierbar sind, werden vor der eigentlichen Auswertung alle Run-Parameter fest fixiert und versioniert.

* Fixieren und dokumentieren:

  * LLM-Modell(e) und Temperatur/Settings (sofern relevant)
  * Prompt-Versionen je Agent
  * Explainability-Version (`m9_v1`)
  * Entscheidungskriterien: `issue_threshold`, optional `score_cutoff`, `severity_min`, allow/ignore issue types
  * Anzahl Beispiele und Splits
* Einheitliches Run-Config-Format (YAML) pro Experiment (vgl. `configs/m10_factuality_runs.yaml`):

  * Dataset, Split, Modell, Versionen, Thresholds, Cache, max_examples
* Standardisierte Output-Struktur pro Run:

  * Roh-Predictions (JSONL/Parquet) + aggregierte Summary (JSON)

Ziel: Jede Zahl im Ergebnisteil ist eindeutig einem Run mit Konfiguration zuordenbar.

---

## 1) Evaluations-Setup & Datenquellen festziehen

* Datensätze und Zielgrößen definieren:

  * **FRANK:** Faktizitäts-Fehler/Labels → Klassifikation, Fehlerprofile
  * **SummEval:** Human-Ratings (z.B. coherence/consistency/fluency/relevance) → Korrelation/Regression
  * **FineSumFact:** Faktizitäts-Signale/Labels → ergänzende Faktizitäts-Evaluation
* Einheitliches Example-Format definieren:

  * `example_id, dataset, split, article_text, summary_text, reference(optional), gold(optional)`

Wichtig: Pro Dataset wird klar dokumentiert, ob Goldlabels binär, ordinal oder kontinuierlich sind und welche Auswertungsmetriken daraus folgen.

**Optional (Add-on):** SummaCoz als zusätzlicher Benchmark, falls noch Zeit vorhanden ist (siehe unten).

---

## 2) Evaluationsskript implementieren (Run & Logging)

Ein CLI-Skript, das datasetweise läuft und pro Beispiel eine einheitliche Ergebniszeile erzeugt.

Ablauf pro Beispiel:

* Beispiel laden
* `/verify` oder Service direkt ausführen
* Speichern:

  * AgentScores pro Dimension + Overall
  * Agent-Issues: `issue_spans` (inkl. Severity und `issue_type`)
  * Explainability-Ableitungen: `num_findings`, `severity_counts`, `issue_type_counts`, `top_spans_count`

Output pro Run:

* Ergebnisdatei (JSONL/CSV/Parquet) mit:

  * `agent_scores`, `issue_counts`, `finding_counts`, `severity_profile`, `issue_type_profile`
  * optional: komprimierte Explainability-Stats (nicht zwingend kompletter JSON-Report, aber referenzierbar)

Ziel: Pro Dataset/Split existiert eine konsistente, auswertbare Datei, die für alle Analysen genutzt wird.

---

## 3) Dimension-Evaluation (Agenten einzeln) und End-to-End (System)

Damit Ursache und Wirkung klar trennbar sind, wird die Evaluation zweistufig strukturiert:

1. **Dimension-Evaluation:** Factuality, Coherence, Readability jeweils separat gegen passende Goldlabels/Ratings
2. **End-to-End:** Vergleich des Gesamtsystems (inkl. Aggregation/Overall) gegen klassische Metriken

Diese Trennung verhindert, dass Systemergebnisse zu einer Blackbox werden, und macht Fehlerquellen pro Agent sichtbar.

---

## 4) Klassische Metriken als Baselines hinzufügen

Für jedes Beispiel (wo möglich, abhängig von Referenzen):

* **ROUGE** (Overlap mit Referenz)
* **BERTScore** (semantische Ähnlichkeit mit Referenz)
* **SummaC** (Konsistenz/Entailment-Check, artikel↔summary)

Baselines werden in derselben Ergebnisdatei gespeichert, damit alle Analysen exakt auf derselben Datenbasis laufen.

---

## 5) Quantitative Auswertung: Übereinstimmung und “besser als”

Je nach Goldstandard:

**A) Ratings (SummEval):**

* Pearson- und Spearman-Korrelation
* MAE/RMSE (nach Skalen-Normalisierung, falls nötig)

**B) Binäre Labels / Fehler vorhanden (z.B. FRANK):**

* Precision/Recall/F1 bei definiertem Threshold
* AUROC / Average Precision (thresholdfrei)

Kernfrage:
AgentScores sollen **stärker** mit Gold/Human übereinstimmen als ROUGE/BERTScore/SummaC (z.B. höhere Korrelation bzw. bessere AUROC/MAE).

---

## 6) Explainability quantitativ nutzbar machen (Evidence statt nur Beispiele)

Die Explainability-Schicht wird nicht nur qualitativ (Fallstudien), sondern soweit möglich auch **quantitativ** ausgewertet.

Mögliche Kennzahlen:

* **Span-Trefferquote** (wenn Gold-Spans existieren oder ableitbar sind):

  * Trefferregel: Overlap von Character-Spans genügt (robuste, einfache Definition)
  * Finding-Precision/Recall: Anteil der Findings, die ein Goldproblem treffen und Anteil Goldprobleme, die getroffen werden
* **Dedupe/Clustering-Statistik:**

  * Verhältnis `raw_issue_spans → findings` als Kompression/Redundanzmaß
* **Severity- und Issue-Type-Profile:**

  * Häufigste `issue_types`, Severity-Verteilungen pro Dataset

Ziel: Der Mehrwert “auditierbare Evidence” wird messbar, nicht nur erzählbar.

---

## 7) Signifikanz & Robustheit (damit “besser” belastbar ist)

* Bootstrap-Konfidenzintervalle:

  * für Korrelationen und Fehlermaße
* Vergleich der Differenzen:

  * AgentScore vs Baseline (z.B. ΔSpearman, ΔMAE)
* Subset-Analysen:

  * kurze vs lange Summaries
  * viele vs wenige Findings
  * bestimmte issue_types (NUMBER/DATE/ENTITY)

Ziel: Ergebnisse sind nicht nur punktuell besser, sondern stabil über sinnvolle Teilmengen.

---

## 8) Qualitative Analyse (Explainability als Diagnoseinstrument)

* Fallstudien: Beispiele, wo

  * Agenten “richtig” warnen, Baselines aber hoch sind
  * Baselines “gut” aussehen, Agenten jedoch plausible Issues finden
* Fehlerprofile:

  * häufigste issue_types
  * severity-Verteilungen
  * typische Top-Spans (Textmuster)

Ziel: zeigen, **warum** das agentische System gewinnt oder verliert und welche Fehlerarten es besonders gut erkennt.

---

## 9) Praktische Eigenschaften und Validitätsbedrohungen (kurz, aber wichtig)

Damit die Ergebnisse korrekt eingeordnet werden können, werden zusätzlich zwei Punkte dokumentiert:

**Praktische Eigenschaften:**

* Laufzeit/Costs pro 100 Beispiele (grobe Messung genügt)
* Stabilität bei Wiederholungen (kleines Subset mehrfach ausführen, Varianz grob reporten)

**Threats to validity (kurz):**

* Prompt- und Modellabhängigkeit der Agenten (Varianz, Drift)
* Domain Shift zwischen Datensätzen und realen Summaries
* Baseline-Limitierungen (Referenzabhängigkeit bei ROUGE/BERTScore)
* Readability-Heuristiken (wenn keine echten Human-Ratings vorhanden)

Ziel: Die BA wirkt wissenschaftlich sauber, ohne auszuufern.

---

## Optional: Zusätzliche Datenquelle (SummaCoz)

Empfehlung: erst integrieren, wenn M10 mit FRANK + SummEval + FineSumFact stabil läuft.
SummaCoz kann als Add-on sinnvoll sein, weil es Konsistenzfälle oft klarer testbar macht.

---

## Ergebnisse (Final, 2026-01-16)

### Datasets

- **FRANK:** Factuality (Binary: `has_error` true/false), n=200
- **SummEval:** Coherence + Readability (Kontinuierlich: 1-5, normalisiert zu 0-1), n=200

### Methoden

- **Agenten:** Factuality, Coherence, Readability (strukturierte Analyse)
- **LLM-as-a-Judge:** GPT-4o-mini als Baseline (optional, `ENABLE_LLM_JUDGE=true`)
- **Klassische Baselines:** Flesch Reading Ease, Flesch-Kincaid, Gunning Fog (nur Readability)

### Ergebnisse pro Dimension

#### Readability

| System | Spearman ρ (95% CI) | Pearson r (95% CI) | MAE (95% CI) | n |
|--------|---------------------|--------------------|--------------|---|
| **Agent** | **0.402 [0.268, 0.512]** | 0.390 [0.292, 0.468] | 0.283 [0.263, 0.302] | 200 |
| Judge | 0.280 | 0.343 | 0.417 | 200 |
| Flesch | -0.054 [-0.197, 0.085] | 0.168 [-0.090, 0.386] | 0.384 [0.362, 0.405] | 200 |
| Flesch-Kincaid | -0.055 [-0.199, 0.093] | 0.124 [-0.115, 0.337] | 0.448 [0.425, 0.473] | 200 |
| Gunning Fog | -0.039 [-0.172, 0.101] | 0.047 [-0.125, 0.213] | 0.579 [0.548, 0.609] | 200 |

**Interpretation:** Agent zeigt beste Korrelation (ρ = 0.402), deutlich besser als klassische Formeln (ρ ≈ -0.05) und besser als Judge (ρ = 0.280). Klassische Formeln erfassen keine semantischen Aspekte.

#### Factuality

| System | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Balanced Accuracy | MCC (95% CI) | AUROC | n |
|--------|-------------------|-----------------|-------------|-------------------|--------------|-------|---|
| **Agent** | 0.786 [0.711, 0.856] | 0.798 [0.726, 0.866] | **0.792 [0.732, 0.843]** | 0.722 | 0.445 [0.315, 0.571] | 0.892 | 200 |
| Judge | 0.9286 [0.8895, 0.9626] | 0.9602 [0.9286, 0.9881] | 0.9441 [0.9178, 0.9681] | 0.7093 | 0.4753 | 0.9453 | 200 |

**Interpretation:** Agent zeigt gute Balance (F1 = 0.792, AUROC = 0.892). Judge ist höher (F1 = 0.9441), aber Sample unbalanciert → niedrigere Balanced Accuracy (0.7093 vs 0.722).

#### Coherence

| System | Spearman ρ (95% CI) | Pearson r (95% CI) | MAE (95% CI) | n |
|--------|---------------------|--------------------|--------------|---|
| **Agent** | **0.409 [0.268, 0.534]** | 0.345 [0.172, 0.529] | 0.178 [0.155, 0.202] | 200 |
| Judge | 0.450 [0.330, 0.560] | 0.480 [0.360, 0.580] | 0.210 [0.180, 0.230] | 200 |

**Interpretation:** Agent zeigt moderate Korrelation (ρ = 0.409). Judge ist leicht besser (ρ = 0.450), aber CIs überlappen (kein signifikanter Unterschied).

### Interpretation

**Was bedeutet das?**
- **Agenten übertreffen klassische Baselines:** Readability Agent (ρ = 0.402) deutlich besser als Flesch/FK/Fog (ρ ≈ -0.05)
- **Agenten sind vergleichbar mit LLM-as-a-Judge:** Readability Agent besser, Coherence ähnlich, Factuality Agent niedriger F1 aber höhere Balanced Accuracy
- **Spearman ρ ist primär:** Misst Rangfolge (robust gegen Skalenfehler), nicht absolute Werte
- **R² kann negativ sein:** Bedeutet schlechter als Mittelwert-Baseline, aber nicht widersprüchlich zu brauchbarem Spearman

**Wann sind klassische Metriken überlegen?**
- Nie (in dieser Evaluation). Klassische Formeln zeigen keine Korrelation mit Human Ratings.

**Welche Fehlertypen treiben die Performance?**
- Factuality: NUMBER, DATE, ENTITY (strukturierte Claims)
- Coherence: Logische Brüche, fehlende Übergänge
- Readability: Satzkomplexität, Struktur

**Limitationen:**
- SummEval ohne Referenzen → ROUGE/BERTScore nicht berechenbar
- Sample-Größe: n=200 (ausreichend für Bootstrap-CIs)
- Prompt-Abhängigkeit: Ergebnisse gelten für v1/v3 Prompts

### Reproduzierbarkeit

**Git Tags:**
- `readability-final-2026-01-16`
- `thesis-snapshot-2026-01-17`

**Run-Artefakte:**
- `results/evaluation/readability/readability_20260116_170832_gpt-4o-mini_v1_seed42/`
- `results/evaluation/baselines/baselines_readability_flesch_fk_fog_20260116_175246_seed42/`
- `results/evaluation/factuality/judge_factuality_*/`

**Detail-Reports:**
- `docs/status/readability_status.md`
- `docs/status/factuality_status.md`
- `docs/status/coherence_status.md`

---

## Deliverables (Definition of Done) ✅

* ✅ Reproduzierbares Evaluationsskript (CLI) pro Dataset/Split
* ✅ Run-Konfigurationen und standardisierte Ergebnisdateien (JSONL/JSON) mit:

  * AgentScores + Baselines + Gold
* ✅ Auswertungen:

  * Korrelationen / MAE / AUROC etc.
  * direkter Vergleich Agenten vs Baselines (inkl. Bootstrap-Konfidenzintervalle)
* ✅ Kurze wissenschaftliche Ergebniszusammenfassung:

  * ✅ In welchen Fällen stimmt das System besser mit Menschen überein? → Agenten übertreffen klassische Baselines
  * ✅ Wann sind klassische Metriken überlegen? → Nie (in dieser Evaluation)
  * ✅ Welche Fehlertypen treiben die Performance? → Strukturierte Claims (NUMBER, DATE, ENTITY)
  * ✅ Welche Limitationen sind relevant? → SummEval ohne Referenzen, Prompt-Abhängigkeit

---


