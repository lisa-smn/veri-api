# ReadabilityAgent Entry Point + Config

**Datum:** 2026-01-08

## Agent Entry Point

**Klasse:** `app/services/agents/readability/readability_agent.py::ReadabilityAgent`

**Pipeline-Integration:**
- `app/pipeline/verification_pipeline.py` (Zeile 37): `self.readability_agent = ReadabilityAgent(self.llm_client)`
- `app/pipeline/verification_pipeline.py` (Zeile 44): `readability = self.readability_agent.run(article, summary, meta)`

**Rückgabeformat:**
- `AgentResult` mit:
  - `name: "readability"`
  - `score: float` (0.0-1.0)
  - `explanation: str`
  - `issue_spans: List[IssueSpan]` (mit `start_char`, `end_char`, `message`, `severity`, optional `issue_type`)
  - `details: Dict` (enthält `issues: List[ReadabilityIssue]`, `num_issues: int`)

## Config Flags / Thresholds

**`ISSUE_FALLBACK_THRESHOLD = 0.7`** (Zeile 28):
- Wenn Score < 0.7 und keine Issues vorhanden sind, werden Fallback-Issues aus Heuristiken erzeugt.
- **Zweck:** Sicherstellen, dass schlechte Scores auch Issues haben (für Explainability).

**Evaluator:**
- Standard: `LLMReadabilityEvaluator(llm_client)` (Zeile 36)
- Kann überschrieben werden via `evaluator`-Parameter im Constructor.

**Prompt-Version:**
- Wird via Environment-Variable `READABILITY_PROMPT_VERSION` gesteuert (default: `"v1"`).
- Wird im Eval-Script verwendet: `os.getenv("READABILITY_PROMPT_VERSION", "v1")`

## Scope

**Readability bewertet:**
- Lesefluss und Verständlichkeit
- Überlange Sätze
- Unnötige Verschachtelung / zu viele Nebensätze
- Interpunktions-Überladung (z.B. extrem viele Kommata, Klammern)
- Schwer zu parsende Satzkonstruktionen

**Nicht Scope:**
- Fakten (→ FactualityAgent)
- Kohärenz/Logik (→ CoherenceAgent)
- Stil/Tonalität (nicht Teil der aktuellen Evaluation)

