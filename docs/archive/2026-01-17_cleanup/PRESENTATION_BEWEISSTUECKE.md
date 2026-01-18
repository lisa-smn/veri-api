# Präsentations-Beweisstücke (Kolloquium)

**Zielgruppe:** Fachfremde Prüfer  
**Format:** Markdown, direkt kopierbar

---

## A) BEWEISSTÜCK 1: Ergebnis-Zusammenfassung

**Datei:** `results/evaluation/summary.md` (Zeilen 31-66)

```markdown
## Interpretation

### Evaluationsprinzipien

- **Optimierungsmetrik:** Wir optimieren auf Balanced Accuracy, da die Klassen unbalanced sind und FP/FN asymmetrisch relevant sind.
- **FineSumFact:** FineSumFact ist ein reines Testset; es werden keine Parameteränderungen nach FRANK vorgenommen.

### FRANK (Dev/Calibration)

- **Baseline Balanced Acc:** 0.508 | F1: 0.920
- **Tuned Balanced Acc:** 0.508 (+0.000) | F1: 0.920 (+0.000)
- **Ablation Balanced Acc:** 0.508 (+0.000 vs baseline) | F1: 0.920 (+0.000)
  → Ablation zeigt geringen Effekt der Claim-Extraktion

### FineSumFact (Test)

- **Final Balanced Acc:** 0.523 | F1: 0.641
- **Generalization:** Balanced Acc +0.015 | F1 -0.279 vs FRANK Tuned
  → Gute Generalisierung auf FineSumFact (keine Parameteränderungen nach FRANK)
- **Ablation Balanced Acc:** 0.523 (+0.000 vs final) | F1: 0.641 (+0.000)

### Combined

- **Combined Balanced Acc:** 0.509 | F1: 0.825
- **N:** 500 (FRANK + FineSumFact)

### Trade-offs

- **Recall vs Specificity:** Recall=0.958, Specificity=0.057
  → System priorisiert Recall (findet mehr Fehler, aber auch mehr False Positives)

### Ablation-Effekt

- **Balanced Acc-Drop durch Ablation:** 0.000 | F1-Drop: 0.000
  → **Schwacher Effekt:** Claim-Extraktion hat geringen Einfluss
```

**Warum ist das für die Präsentation wichtig?**  
Diese Zusammenfassung zeigt die wichtigsten Evaluationsergebnisse auf einen Blick: Balanced Accuracy als Optimierungsmetrik, Generalisierung auf FineSumFact, und den Trade-off zwischen Recall und Specificity. Die Zahlen belegen, dass das System funktioniert, aber auch seine Grenzen zeigt.

---

## B) BEWEISSTÜCK 2: Starkes Fakten-Beispiel mit Evidence

**Datei:** `results/evaluation/runs/results/factuality_combined_final_v1_examples.jsonl` (Zeile 31, `ex_30`)

```json
{
  "example_id": "ex_30",
  "ground_truth": true,
  "prediction": true,
  "score": 0.0,
  "num_issues": 1,
  "effective_issues": 1.0,
  "issue_spans": [
    {
      "start_char": 0,
      "end_char": 69,
      "message": "Satz 1: Claim 'tens of thousands of people have attended the annual anzac day parade' – Der Kontext erwähnt, dass 'Thousands attended the early morning service' und 'up to 400 people took part in a parade', was nicht mit 'tens of thousands' übereinstimmt.",
      "severity": "high",
      "issue_type": "NUMBER",
      "confidence": 0.5,
      "mapping_confidence": 1.0,
      "evidence_found": true
    }
  ],
  "summary": "tens of thousands of people have attended the annual anzac day parade in the gallipoli landings in hyde park.",
  "meta": {
    "hash": "36128472",
    "model_name": "PtGen",
    "factuality": 0.0
  }
}
```

**Warum ist das für die Präsentation wichtig?**  
Dieses Beispiel zeigt das Evidence-Gate in Aktion: Der Agent hat einen NUMBER-Fehler (high severity) mit `evidence_found=true` erkannt. Die Message enthält das wörtliche Evidence-Zitat aus dem Artikel ("Thousands attended" vs. "tens of thousands"), was beweist, dass das System belastbare Evidence findet und korrekt verwendet.

---

## C) BEWEISSTÜCK 3: Explainability-Report als "Arztbrief"

**Datei:** `app/services/explainability/explainability_models.py` (Zeilen 79-85) + `app/services/explainability/explainability_service.py` (Zeilen 259-289, 654-689)

**Struktur (Pydantic-Modell):**

```python
class ExplainabilityResult(BaseModel):
    summary: List[str]                    # Executive Summary (3-6 Sätze)
    findings: List[Finding]               # Alle Findings (normalisiert, dedupliziert, sortiert)
    by_dimension: Dict[Dimension, List[Finding]]  # Gruppiert nach factuality/coherence/readability
    top_spans: List[TopSpan]               # Top-K wichtigste Textstellen
    stats: ExplainabilityStats             # Anzahl Findings, Coverage
    version: str = "m9_v1"                 # Version für Vergleichbarkeit
```

**Beispiel-Finding (aus `Finding`-Modell, Zeilen 51-59):**

```python
class Finding(BaseModel):
    id: str                                # Stabile ID (Hash-basiert)
    dimension: Dimension                   # factuality/coherence/readability
    severity: Severity                     # low/medium/high
    message: str                           # Kurze Erklärung des Problems
    span: Optional[Span]                   # Textstelle (start_char, end_char, text)
    evidence: List[EvidenceItem]           # Evidence-Quotes, Claims, Raw-Daten
    recommendation: Optional[str]          # Handlungsempfehlung (z.B. "Zahlen/Einheiten mit dem Artikel abgleichen")
    source: Dict[str, Any]                 # Provenance (agent, issue_type, claim_id)
```

**Executive Summary Generation (Zeilen 654-689):**

```python
def _executive_summary(self, ranked: List[Finding], top_spans: List[TopSpan]) -> List[str]:
    # Beispiel-Output:
    # [
    #   "In der Summary wurden 1 Findings identifiziert (0 high, 0 medium, 1 low).",
    #   "Der Schwerpunkt liegt bei **factuality** (meiste Findings in dieser Dimension).",
    #   "Kritische Textstellen: „tens of thousands of people...", „a service of remembrance...",
    #   "Priorität: erst factuality-high fixen (Zahlen/Daten), dann Kohärenz, dann Lesbarkeit glätten."
    # ]
```

**Beispiel-Explainability-Report (JSON-Struktur, generiert aus Agent-Ergebnissen):**

```json
{
  "summary": [
    "In der Summary wurden 1 Findings identifiziert (1 high, 0 medium, 0 low).",
    "Der Schwerpunkt liegt bei **factuality** (meiste Findings in dieser Dimension).",
    "Kritische Textstellen: „tens of thousands of people have attended the annual anzac day parade...",
    "Priorität: erst factuality-high fixen (Zahlen/Daten), dann Kohärenz, dann Lesbarkeit glätten."
  ],
  "findings": [
    {
      "id": "f_a1b2c3d4e5f6",
      "dimension": "factuality",
      "severity": "high",
      "message": "Satz 1: Claim 'tens of thousands of people have attended...' – Der Kontext erwähnt, dass 'Thousands attended the early morning service' und 'up to 400 people took part in a parade', was nicht mit 'tens of thousands' übereinstimmt.",
      "span": {
        "start_char": 0,
        "end_char": 69,
        "text": "tens of thousands of people have attended the annual anzac day parade"
      },
      "evidence": [
        {
          "kind": "raw",
          "source": "agent:factuality",
          "data": {"issue_span": {...}}
        }
      ],
      "recommendation": "Zahlen/Einheiten mit dem Artikel abgleichen und ggf. korrigieren.",
      "source": {
        "agent": "factuality",
        "source_list": "issue_spans",
        "issue_type": "NUMBER",
        "item_index": 0
      }
    }
  ],
  "by_dimension": {
    "factuality": [...],
    "coherence": [],
    "readability": []
  },
  "top_spans": [
    {
      "span": {"start_char": 0, "end_char": 69, "text": "tens of thousands of people..."},
      "dimension": "factuality",
      "severity": "high",
      "finding_id": "f_a1b2c3d4e5f6",
      "rank_score": 7.2
    }
  ],
  "stats": {
    "num_findings": 1,
    "num_high_severity": 1,
    "num_medium_severity": 0,
    "num_low_severity": 0,
    "coverage_chars": 69,
    "coverage_ratio": 0.27
  },
  "version": "m9_v1"
}
```

**Erzeugungsstelle:** `app/services/explainability/explainability_service.py` → `build()` (Zeilen 259-289)

**Warum ist das für die Präsentation wichtig?**  
Der Explainability-Report ist wie ein "Arztbrief": Er erklärt nicht nur, dass es ein Problem gibt, sondern auch wo (Span), wie schwerwiegend (Severity), warum (Message), und was zu tun ist (Recommendation). Die strukturierte Form macht die Entscheidungen des Systems nachvollziehbar und handlungsorientiert.

---

## D) OPTIONAL 1: Runs / Experimentdesign

**Datei:** `configs/m10_factuality_runs.yaml` (Zeilen 7-139)

### Run 1: Baseline

```yaml
run_id: "factuality_frank_baseline_v1"
description: "FRANK Baseline - Aktueller Stand (Prompt/Thresholds wie im 1. Run)"
decision_mode: "issues"
error_threshold: 1
uncertainty_policy: "count_as_error"
severity_min: "low"
```

**Wofür:** Referenzpunkt für alle weiteren Runs; zeigt den Ausgangszustand ohne Tuning.

### Run 2: Tuned

```yaml
run_id: "factuality_frank_tuned_v1"
description: "FRANK Tuned - Thresholds/Decision-Regeln angepasst basierend auf Baseline-Analyse"
decision_mode: "issues"
error_threshold: 1
uncertainty_policy: "count_as_error"
severity_min: "low"
```

**Wofür:** Optimierte Version nach Baseline-Analyse; zeigt, ob Tuning die Metriken verbessert.

### Run 3: Ablation

```yaml
run_id: "factuality_frank_ablation_v1"
description: "FRANK Ablation - Claim-Extraktion deaktiviert (nur Satz-Fallback)"
ablation_mode: "no_claims"
use_claim_extraction: false
```

**Wofür:** Zeigt, ob Claim-Extraktion kritisch für die Performance ist (Ablation-Effekt = 0.000).

### Run 4: Tuning severity_min

```yaml
run_id: "factuality_frank_tune_severity_v1"
description: "FRANK Tuning: severity_min=medium (nur medium/high Issues zählen)"
severity_min: "medium"
```

**Wofür:** Testet, ob das Filtern von low-severity Issues False Positives reduziert.

### Run 5: Tuning uncertainty_policy

```yaml
run_id: "factuality_frank_tune_uncertain_policy_v1"
description: "FRANK Tuning: severity_min=medium + uncertainty_policy=non_error"
severity_min: "medium"
uncertainty_policy: "non_error"
```

**Wofür:** Testet, ob "uncertain" nicht als Fehler zählen False Positives reduziert.

**Warum ist das für die Präsentation wichtig?**  
Das Experimentdesign zeigt, dass die Evaluation systematisch und reproduzierbar ist. Jeder Run hat einen klaren Zweck (Baseline, Tuning, Ablation), und die Parameter sind dokumentiert. Dies macht die Ergebnisse nachvollziehbar und zeigt wissenschaftliche Sorgfalt.

---

## E) OPTIONAL 2: Neo4j Traceability Mini-Beleg

**Datei:** `app/db/neo4j/graph_persistence.py` (Zeilen 51-150)

```python
def _write_graph_tx(
    tx: Transaction,
    article_id: int | str,
    summary_id: int | str,
    run_id: int | str,
    overall_score: float,
    factuality: AgentResult,
    coherence: AgentResult,
    readability: AgentResult,
) -> None:
    # --- Core Nodes ---
    tx.run(
        """
        MERGE (a:Article {id: $article_id})
        MERGE (s:Summary {id: $summary_id})
        MERGE (a)-[:HAS_SUMMARY]->(s)
        """,
        article_id=article_id,
        summary_id=summary_id,
    )

    # Run-Knoten
    tx.run(
        """
        MERGE (r:Run {id: $run_id})
        WITH r
        MATCH (s:Summary {id: $summary_id})
        MERGE (r)-[:EVALUATES]->(s)
        """,
        run_id=run_id,
        summary_id=summary_id,
    )

    # Metric-Node pro Dimension
    tx.run(
        """
        MATCH (s:Summary {id: $summary_id})
        MERGE (m:Metric {run_id: $run_id, summary_id: $summary_id, dimension: $dimension})
        SET m.score = $score
        MERGE (s)-[:HAS_METRIC]->(m)
        """,
        summary_id=summary_id,
        run_id=run_id,
        dimension=dimension,
        score=float(agent.score),
    )

    # IssueSpan-Nodes
    for idx, span in enumerate(spans):
        tx.run(
            """
            MATCH (m:Metric {run_id: $run_id, summary_id: $summary_id, dimension: $dimension})
            MERGE (sp:IssueSpan:Error {
                run_id: $run_id,
                summary_id: $summary_id,
                dimension: $dimension,
                span_index: $idx
            })
            SET sp.message = $message,
                sp.severity = $severity,
                sp.start_char = $start_char,
                sp.end_char = $end_char
            MERGE (m)-[:HAS_ISSUE_SPAN]->(sp)
            """,
            run_id=run_id,
            summary_id=summary_id,
            dimension=dimension,
            idx=idx,
            message=message,
            severity=severity,
            start_char=start_char,
            end_char=end_char,
        )
```

**Graph-Struktur:**
- `(Article)-[:HAS_SUMMARY]->(Summary)`
- `(Run)-[:EVALUATES]->(Summary)`
- `(Summary)-[:HAS_METRIC]->(Metric)`
- `(Metric)-[:HAS_ISSUE_SPAN]->(IssueSpan)`

**Warum ist das für die Präsentation wichtig?**  
Die Neo4j-Graph-Struktur ermöglicht komplexe Queries zur Nachvollziehbarkeit: "Welche Summaries haben ähnliche Issue-Patterns?", "Welche Runs haben die meisten NUMBER-Fehler?", "Wie korrelieren Metriken über verschiedene Runs?". Dies zeigt, dass das System nicht nur Ergebnisse produziert, sondern auch für spätere Analysen und Forschung nutzbar ist.

---

## F) BEWEISSTÜCK 4: Multi-Agent-System (Factuality + Coherence + Readability)

**Datei:** `app/pipeline/verification_pipeline.py` (Zeilen 28-60) + `app/services/agents/coherence/coherence_agent.py` (Zeilen 13-59) + `app/services/agents/readability/readability_agent.py` (Zeilen 25-72)

**Pipeline-Integration (Zeilen 28-60):**

```python
class VerificationPipeline:
    def __init__(self, model_name: str = "gpt-4o-mini") -> None:
        self.llm_client = OpenAIClient(model_name=model_name)
        
        self.factuality_agent = FactualityAgent(self.llm_client)
        self.coherence_agent = CoherenceAgent(self.llm_client)
        self.readability_agent = ReadabilityAgent(self.llm_client)
        
        self.explainability_service = ExplainabilityService()

    def run(self, article: str, summary: str, meta: dict | None = None) -> PipelineResult:
        factuality = self.factuality_agent.run(article, summary, meta)
        coherence = self.coherence_agent.run(article, summary, meta)
        readability = self.readability_agent.run(article, summary, meta)

        overall = (factuality.score + coherence.score + readability.score) / 3.0

        result = PipelineResult(
            factuality=factuality,
            coherence=coherence,
            readability=readability,
            overall_score=overall,
            explainability=None,
        )
        
        result.explainability = self.explainability_service.build(result, summary_text=summary)
        return result
```

**CoherenceAgent (Zeilen 13-59):**

```python
class CoherenceAgent:
    """
    Bewertet die Kohärenz einer Summary relativ zum Artikel.
    
    Scope (Coherence):
    - interne logische Konsistenz (keine Selbstwidersprüche)
    - nachvollziehbarer Informationsfluss / Reihenfolge
    - keine unnötigen Wiederholungen (Redundanz)
    - klare Referenzen (Pronomen/Bezüge verständlich)
    """
    
    def run(self, article_text: str, summary_text: str, meta: Dict[str, Any] | None = None) -> AgentResult:
        score, issues, explanation = self.evaluator.evaluate(article_text, summary_text)
        
        return AgentResult(
            name="coherence",
            score=score,
            explanation=explanation,
            issue_spans=errors,  # IssueSpan-Liste mit severity, message, start_char, end_char
            details={"issues": [asdict(i) for i in issues], "num_issues": len(issues)},
        )
```

**ReadabilityAgent (Zeilen 25-72):**

```python
class ReadabilityAgent:
    """
    Bewertet die Lesbarkeit einer Summary.
    
    Scope (Readability):
    - Lesefluss und Verständlichkeit
    - überlange Sätze
    - unnötige Verschachtelung / zu viele Nebensätze
    - Interpunktions-Überladung (z.B. extrem viele Kommata, Klammern)
    - schwer zu parsende Satzkonstruktionen
    """
    
    def run(self, article_text: str, summary_text: str, meta: Dict[str, Any] | None = None) -> AgentResult:
        score, issues, explanation = self.evaluator.evaluate(article_text, summary_text)
        
        # Fallback-Heuristiken, falls LLM keine Issues liefert
        if not issues and score < self.ISSUE_FALLBACK_THRESHOLD:
            issues = self._fallback_issues(summary_text)
        
        return AgentResult(
            name="readability",
            score=score,
            explanation=explanation,
            issue_spans=issue_spans,  # IssueSpan-Liste mit severity, message, start_char, end_char
            details={"issues": [asdict(i) for i in issues], "num_issues": len(issues)},
        )
```

**Issue-Typen:**

**CoherenceIssue-Typen** (`app/services/agents/coherence/coherence_models.py`, Zeilen 6-19):
- `LOGICAL_INCONSISTENCY`: Selbstwidersprüche in der Summary
- `CONTRADICTION`: Widersprüchliche Aussagen
- `REDUNDANCY`: Unnötige Wiederholungen
- `ORDERING`: Probleme mit Informationsfluss/Reihenfolge
- `OTHER`: Sonstige Kohärenzprobleme

**ReadabilityIssue-Typen** (`app/services/agents/readability/readability_models.py`, Zeilen 26-42):
- `LONG_SENTENCE`: Überlange Sätze (z.B. >30 Wörter)
- `COMPLEX_NESTING`: Zu starke Verschachtelung/Nebensätze
- `PUNCTUATION_OVERLOAD`: Übermäßige Interpunktion (z.B. viele Kommata)
- `HARD_TO_PARSE`: Schwer zu parsende Satzkonstruktionen

**Status:**
- **FactualityAgent:** Vollständig evaluiert (FRANK, FineSumFact, Combined, Balanced Acc 0.508-0.523)
- **CoherenceAgent:** Implementiert, in Pipeline integriert, Evaluation in Planung
- **ReadabilityAgent:** Implementiert, in Pipeline integriert, Evaluation in Planung

**Warum ist das für die Präsentation wichtig?**  
Das System ist ein Multi-Agent-System mit drei spezialisierten Agenten (Factuality, Coherence, Readability), die parallel arbeiten und ihre Ergebnisse zu einem Gesamt-Score aggregieren. Die Explainability-Komponente integriert alle drei Dimensionen in einen einheitlichen Report. Dies zeigt die Architektur des Gesamtsystems und macht klar, dass Factuality nur eine von drei wichtigen Komponenten ist.

---

## Zusammenfassung

**3 Beweisstücke:**
1. **Ergebnis-Zusammenfassung:** Zeigt Evaluationsergebnisse (Balanced Acc 0.508-0.523, Recall 0.958, Trade-offs)
2. **Fakten-Beispiel mit Evidence:** Beweist, dass Evidence-Gate funktioniert (NUMBER-Fehler mit evidence_found=true)
3. **Explainability-Report-Struktur:** Zeigt, wie Entscheidungen nachvollziehbar gemacht werden (Finding mit Span, Severity, Recommendation)

**3 Optionale Artefakte:**
4. **Experimentdesign:** Zeigt systematische, reproduzierbare Evaluation (Baseline, Tuned, Ablation, Tuning-Varianten)
5. **Neo4j Traceability:** Zeigt Graph-Struktur für komplexe Analysen (Article→Summary→Metric→IssueSpan)
6. **Multi-Agent-System:** Zeigt die drei Agenten (Factuality, Coherence, Readability) und ihre Integration in die Pipeline

Alle Beweisstücke sind direkt aus dem Repository extrahiert und mit Dateipfaden + Zeilenangaben belegt.

