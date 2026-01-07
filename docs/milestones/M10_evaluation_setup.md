
---

# M10 – Evaluation & Vergleich mit klassischen Metriken

## Ziel

In M10 wird das System **systematisch und reproduzierbar evaluiert**.
Die Agent-Scores (Factuality/Coherence/Readability) sowie der Explainability-Report werden **quantitativ** gegen Goldlabels bzw. Human-Ratings geprüft und **direkt** mit klassischen Metriken (z.B. ROUGE, BERTScore, SummaC) verglichen.

Ergebnis sind belastbare Aussagen darüber, **ob** und **wann** das agentische System besser mit menschlichen Bewertungen übereinstimmt als klassische Metriken und welchen Mehrwert die Explainability-Schicht für nachvollziehbare Fehlerdiagnosen liefert.

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
* Einheitliches Run-Config-Format (JSON/YAML) pro Experiment:

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

## Deliverables (Definition of Done)

* Reproduzierbares Evaluationsskript (CLI) pro Dataset/Split
* Run-Konfigurationen (JSON/YAML) und standardisierte Ergebnisdateien (JSONL/CSV/Parquet) mit:

  * AgentScores + Explainability-Stats + Baselines + Gold
* Auswertungen:

  * Korrelationen / MAE / AUROC etc.
  * direkter Vergleich Agenten vs ROUGE/BERTScore/SummaC (inkl. Konfidenzintervalle)
  * Subset- und Fehlerprofil-Analysen
* Qualitative Fallstudien mit Evidence-Spans aus Explainability
* Kurze wissenschaftliche Ergebniszusammenfassung:

  * In welchen Fällen stimmt das System besser mit Menschen überein?
  * Wann sind klassische Metriken überlegen?
  * Welche Fehlertypen treiben die Performance?
  * Welche Limitationen sind relevant?

---


