# Kapitel 5 – Implementierungsdetails (Vervollständigung)

**Datum:** 2026-01-17  
**Zweck:** Repo-basierte Dokumentation der fehlenden Implementierungsdetails für Kapitel 5

---

## I) Prompts (vollständig)

| Prompt-Name | Dimension/Use-Case | Datei+Zeilenbereich | Variablen/Inputs | Output-Format/Constraints | Versioning |
|-------------|-------------------|---------------------|-------------------|---------------------------|------------|
| **Claim-Extraction** | Factuality (atomare Claims aus Sätzen) | `app/services/agents/factuality/claim_extractor.py:163-205` | `sentence: str` | JSON: `{"claims": [{"text": "..."}]}`, max 5 Claims, Substring-Constraint (Claim muss wörtlich im Satz vorkommen) | Keine Versionierung (einheitlicher Prompt) |
| **Claim-Verification (mit Evidence-Liste)** | Factuality (Verifikation gegen Evidence-Passagen) | `app/services/agents/factuality/claim_verifier.py:448-502` | `evidence_context_list: list[str]`, `claim_text: str` | JSON: `{"label": "correct"\|"incorrect"\|"uncertain", "confidence": float, "error_type": "ENTITY"\|"NUMBER"\|"DATE"\|"OTHER"\|null, "explanation": str, "selected_evidence_index": int, "evidence_quote": str\|null}` | Keine Versionierung (einheitlicher Prompt) |
| **Claim-Verification (ohne Evidence-Liste)** | Factuality (Verifikation gegen Kontext) | `app/services/agents/factuality/claim_verifier.py:503-540` | `context: str`, `claim_text: str` | JSON: `{"label": "correct"\|"incorrect"\|"uncertain", "confidence": float, "error_type": "ENTITY"\|"NUMBER"\|"DATE"\|"OTHER"\|null, "explanation": str, "selected_evidence_index": -1, "evidence_quote": null}` | Keine Versionierung (einheitlicher Prompt) |
| **Coherence Agent** | Coherence (logische Konsistenz, Satzstruktur) | `app/services/agents/coherence/coherence_verifier.py:117-160` | `article: str`, `summary: str` | JSON: `{"score": float [0,1], "explanation": str, "issues": [{"type": "LOGICAL_INCONSISTENCY"\|"CONTRADICTION"\|"REDUNDANCY"\|"ORDERING"\|"OTHER", "severity": "low"\|"medium"\|"high", "summary_span": str, "comment": str, "hint": str\|null}]}`, max 8 Issues | Keine Versionierung (einheitlicher Prompt) |
| **Readability Agent v1** | Readability (Lesbarkeit, Satzkomplexität) | `app/services/agents/readability/readability_verifier.py:159-206` | `article: str`, `summary: str` | JSON: `{"score": float [0,1], "explanation": str, "issues": [{"type": "LONG_SENTENCE"\|"COMPLEX_NESTING"\|"PUNCTUATION_OVERLOAD"\|"HARD_TO_PARSE", "severity": "low"\|"medium"\|"high", "summary_span": str, "comment": str, "metric": str\|null, "metric_value": float\|null}]}`, max 8 Issues | `prompt_version="v1"` (Default) |
| **Readability Agent v2** | Readability (Lesbarkeit, 1-5 Rating) | `app/services/agents/readability/readability_verifier.py:208-280` | `article: str`, `summary: str` | JSON: `{"score": int [1,5], "explanation": str, "issues": [...]}` (gleiche Issue-Struktur wie v1) | `prompt_version="v2"` |
| **LLM-as-a-Judge: Readability v1** | Readability (Judge-Baseline, 1-5 Rating) | `app/services/judges/prompts.py:26-56` | `summary_text: str`, `article_text: str\|None` | JSON: `{"rating": int [1,5], "confidence": float [0,1], "rationale": str}` | `prompt_version="v1"` (Default) |
| **LLM-as-a-Judge: Readability v2_float** | Readability (Judge-Baseline, 0.00-1.00 Score) | `app/services/judges/prompts.py:58-97` | `summary_text: str`, `article_text: str\|None` | JSON: `{"score": float [0.00,1.00], "confidence": float [0,1], "rationale": str}` | `prompt_version="v2_float"` |
| **LLM-as-a-Judge: Coherence v1** | Coherence (Judge-Baseline, 1-5 Rating) | `app/services/judges/prompts.py:116-145` | `summary_text: str`, `article_text: str\|None` | JSON: `{"rating": int [1,5], "confidence": float [0,1], "rationale": str}` | `prompt_version="v1"` (Default) |
| **LLM-as-a-Judge: Factuality v1** | Factuality (Judge-Baseline, 1-5 Rating) | `app/services/judges/prompts.py:166-196` | `summary_text: str`, `article_text: str` (erforderlich) | JSON: `{"rating": int [1,5], "confidence": float [0,1], "rationale": str}` | `prompt_version="v1"` |
| **LLM-as-a-Judge: Factuality v2_binary** | Factuality (Judge-Baseline, binary verdict) | `app/services/judges/prompts.py:199-239` | `summary_text: str`, `article_text: str` (erforderlich) | JSON: `{"error_present": bool, "confidence": float [0,1], "rationale": str}` | `prompt_version="v2_binary"` (Default für Factuality Judge) |

**Hinweise:**
- **Evidence-Retrieval:** Nicht promptbasiert, sondern deterministisch via Sliding-Window + Jaccard-Similarity (`app/services/agents/factuality/evidence_retriever.py:45-98`)
- **Prompt-Versionierung:** Nur Readability Agent und LLM-as-a-Judge unterstützen Versionierung; Claim-Extraction/Verification und Coherence Agent verwenden einheitliche Prompts
- **YAML-Config:** `run_tag: "v3_uncertain_spans"` in `configs/m10_factuality_runs.yaml:15` ist ein Run-/Cache-Tag für M10-Evaluationsläufe (nicht an Agent-Prompt gekoppelt). Legacy `prompt_version` wird als `run_tag` gemappt.

---

## II) LLM-Parameter/Defaults

| Call-Site (Agent/Service) | Client-Methode | Parameter (explizit) | Defaults | Quelle (Datei+Zeilen) | Bemerkung |
|---------------------------|----------------|----------------------|----------|----------------------|-----------|
| **OpenAIClient.complete()** | `OpenAI.chat.completions.create()` | `model=self.model_name`, `messages=[{"role": "user", "content": prompt}]`, `temperature=0.0`, `max_tokens=800` | `temperature=0.0`, `max_tokens=800` | `app/llm/openai_client.py:17-25` | **Zentrale Defaults:** Alle LLM-Calls verwenden diese Parameter, es gibt keine Agent-spezifischen Overrides |
| **LLM-as-a-Judge** | `OpenAIClient.complete()` | `temperature` via Parameter (Default: 0.0), `max_tokens=800` (aus OpenAIClient) | `temperature=0.0` (Default in `LLMJudge.__init__()`), `max_tokens=800` (aus OpenAIClient) | `app/services/judges/llm_judge.py:40-47`, `app/services/judges/llm_judge.py:85`, `app/llm/openai_client.py:21-22` | **Priorität:** `LLMJudge.default_temperature` (Default: 0.0) kann via `judge_temperature` Parameter überschrieben werden (`app/services/verification_service.py:82-83`) |
| **Factuality Agent (Claim-Extraction/Verification)** | `OpenAIClient.complete()` | `temperature=0.0`, `max_tokens=800` (aus OpenAIClient) | `temperature=0.0`, `max_tokens=800` | `app/llm/openai_client.py:21-22` | Keine Agent-spezifischen Overrides |
| **Coherence Agent** | `OpenAIClient.complete()` | `temperature=0.0`, `max_tokens=800` (aus OpenAIClient) | `temperature=0.0`, `max_tokens=800` | `app/llm/openai_client.py:21-22` | Keine Agent-spezifischen Overrides |
| **Readability Agent** | `OpenAIClient.complete()` | `temperature=0.0`, `max_tokens=800` (aus OpenAIClient) | `temperature=0.0`, `max_tokens=800` | `app/llm/openai_client.py:21-22` | Keine Agent-spezifischen Overrides |

**Parameter-Details:**
- **temperature:** `0.0` (deterministisch, keine Zufälligkeit)
- **max_tokens:** `800` (ausreichend für JSON-Responses)
- **top_p, presence_penalty, frequency_penalty:** Nicht gesetzt (OpenAI-Defaults)
- **response_format:** Nicht gesetzt (JSON wird via Prompt erzwungen, nicht via `response_format="json_object"`)
- **seed:** Nicht gesetzt auf Client-Ebene (jedoch in YAML-Configs: `llm_seed: 42` für Evaluation, `app/services/run_manager.py:35`)
- **retry/backoff:** Nicht explizit implementiert (OpenAI-Client verwendet Standard-Retry-Logik)

**Priorität (wer überschreibt wen):**
1. **OpenAIClient Defaults** (`temperature=0.0`, `max_tokens=800`) gelten für alle Calls
2. **LLM-as-a-Judge:** `LLMJudge.default_temperature` (Default: 0.0) kann via `judge_temperature` Parameter überschrieben werden (`app/services/verification_service.py:82-83`)
3. **YAML-Configs:** `llm_temperature` und `llm_seed` in `configs/m10_factuality_runs.yaml` werden für Evaluation-Runs verwendet, aber nicht direkt an OpenAIClient übergeben (möglicherweise via Run-Manager)

---

## III) Neo4j Graphschema

### III.1 Schema-Tabelle (Nodes)

| Label | Key Props | Weitere Props | Quelle |
|-------|-----------|---------------|--------|
| **Article** | `id: int\|str` (MERGE-Key) | - | `app/db/neo4j/graph_persistence.py:64-66` |
| **Summary** | `id: int\|str` (MERGE-Key) | - | `app/db/neo4j/graph_persistence.py:65-66` |
| **Run** | `id: int\|str` (MERGE-Key) | `run_id: int\|str` (SET) | `app/db/neo4j/graph_persistence.py:73-83` |
| **Metric** | `run_id: int\|str`, `summary_id: int\|str`, `dimension: str` (MERGE-Key, kombiniert) | `score: float` (SET) | `app/db/neo4j/graph_persistence.py:87-98` |
| **IssueSpan** (Dual-Label: `IssueSpan:Error`) | `run_id: int\|str`, `summary_id: int\|str`, `dimension: str`, `span_index: int` (MERGE-Key, kombiniert) | `message: str`, `severity: str\|None`, `start_char: int\|None`, `end_char: int\|None` (alle SET) | `app/db/neo4j/graph_persistence.py:108-134` |

**Hinweise:**
- **MERGE-Strategie:** Nodes werden via `MERGE` erstellt (idempotent, verhindert Duplikate)
- **SET-Strategie:** Properties werden via `SET` aktualisiert (überschreibt vorhandene Werte)
- **Dual-Label:** `IssueSpan:Error` ermöglicht Abfragen sowohl mit `IssueSpan` als auch mit `Error` Label (`app/db/neo4j/graph_persistence.py:113`)

### III.2 Relationship-Tabelle

| Type | From | To | Props | Quelle |
|------|------|-----|--------|--------|
| **HAS_SUMMARY** | Article | Summary | - | `app/db/neo4j/graph_persistence.py:66` |
| **EVALUATES** | Run | Summary | - | `app/db/neo4j/graph_persistence.py:79` |
| **HAS_METRIC** | Summary | Metric | - | `app/db/neo4j/graph_persistence.py:92`, `app/db/neo4j/graph_persistence.py:146` |
| **HAS_ISSUE_SPAN** | Metric | IssueSpan | - | `app/db/neo4j/graph_persistence.py:124` |

**Hinweise:**
- **Richtung:** Alle Relationships sind gerichtet (From → To)
- **Properties:** Keine Relationships haben Properties
- **MERGE-Strategie:** Relationships werden via `MERGE` erstellt (idempotent)

### III.3 Optional: ASCII-Graphdiagramm

```
(Article {id})
    |
    | HAS_SUMMARY
    |
    v
(Summary {id})
    |
    | HAS_METRIC
    |
    v
(Metric {run_id, summary_id, dimension, score})
    |
    | HAS_ISSUE_SPAN
    |
    v
(IssueSpan:Error {run_id, summary_id, dimension, span_index, message, severity, start_char, end_char})

(Run {id, run_id})
    |
    | EVALUATES
    |
    v
(Summary {id})
```

---

## IV) Ablation-Modi

| Name | Zweck | Effekt auf Pipeline/Agent | Implementierung (Datei+Zeile) | Wo konfigurierbar |
|------|-------|---------------------------|-------------------------------|-------------------|
| **use_claim_extraction** | Deaktiviert LLM-basierte Claim-Extraktion | Fallback: ganzer Satz als Claim (`NoOpClaimExtractor`) | `app/services/agents/factuality/factuality_agent.py:50-77` | YAML: `use_claim_extraction: false`, Code: `FactualityAgent(..., use_claim_extraction=False)` |
| **use_claim_verification** | Deaktiviert Claim-Verifikation | Alle Claims als "uncertain" markiert (`NoOpClaimVerifier`) | `app/services/agents/factuality/factuality_agent.py:60-88` | YAML: `use_claim_verification: false`, Code: `FactualityAgent(..., use_claim_verification=False)` |
| **use_spans** | Deaktiviert IssueSpan-Generierung | Keine `issue_spans` im `AgentResult` | `app/services/agents/factuality/factuality_agent.py:62-255` | YAML: `use_spans: false`, Code: `FactualityAgent(..., use_spans=False)` |
| **ablation_mode: "no_claims"** | Kombiniert: `use_claim_extraction=false` | Fallback: ganzer Satz als Claim, keine LLM-Extraktion | `app/services/agents/factuality/factuality_agent.py:74-75`, `configs/m10_factuality_runs.yaml:24` | YAML: `ablation_mode: "no_claims"` |
| **ablation_mode: "sentence_only"** | Satz-basierte Claims ohne LLM | `SentenceOnlyExtractor` gibt ganzen Satz als Claim zurück | `app/services/agents/factuality/ablation_extractor.py:18-35` | Code: `SentenceOnlyExtractor` (nicht direkt in YAML, aber via `use_claim_extraction=false` + Fallback) |
| **ablation_mode: "no_spans"** | Deaktiviert IssueSpan-Generierung | Keine `issue_spans` im `AgentResult` | `app/services/agents/factuality/factuality_agent.py:255` | YAML: `ablation_mode: "no_spans"` |
| **require_evidence_for_correct** | Erzwingt Evidence für "correct"-Labels | "correct" ohne Evidence → "uncertain" (Confidence clamp auf 0.55) | `app/services/agents/factuality/claim_verifier.py:52-344` | Code: `LLMClaimVerifier(..., require_evidence_for_correct=True)` (Default: True) |

**Kurztext: Wie Ablation im System umgesetzt ist**

Ablation-Modi werden im Factuality-Agent über Flags (`use_claim_extraction`, `use_claim_verification`, `use_spans`) und einen kombinierten `ablation_mode` Parameter gesteuert. Die Implementierung nutzt **Strategy-Pattern**: Bei deaktivierter Claim-Extraktion wird `NoOpClaimExtractor` verwendet, der eine leere Liste zurückgibt und damit den Fallback (ganzer Satz als Claim) auslöst. Bei deaktivierter Claim-Verifikation wird `NoOpClaimVerifier` verwendet, der alle Claims als "uncertain" markiert. Die Ablation-Modi sind in YAML-Configs (`configs/m10_factuality_runs.yaml`) konfigurierbar und werden für Evaluation-Runs verwendet, um den Einfluss einzelner Komponenten zu messen. Die Implementierung ist **nicht destruktiv**: Deaktivierte Komponenten werden durch No-Op-Implementierungen ersetzt, sodass die Pipeline weiterhin funktioniert, aber bestimmte Schritte übersprungen werden.

---

## V) Doku-Patchvorschläge

### V.1 `docs/status/kapitel5_implementierung_backbone.md`

**Überschrift:** Nach Abschnitt "E) Fehlende Infos/ToDos" ergänzen:

**Neuer Abschnitt: "F) Implementierungsdetails (Vervollständigt)"**

**Bulletpoints:**
- **Prompts:** Alle Prompt-Texte dokumentiert (Claim-Extraction, Claim-Verification, Coherence, Readability, LLM-as-a-Judge), inkl. Versionierung und Output-Formate → `docs/status/kapitel5_implementierungsdetails.md` (Abschnitt I)
- **LLM-Parameter:** Zentrale Defaults dokumentiert (`temperature=0.0`, `max_tokens=800`), Priorität zwischen Client-Defaults und Agent-Overrides geklärt → `docs/status/kapitel5_implementierungsdetails.md` (Abschnitt II)
- **Neo4j Schema:** Vollständiges Graphschema dokumentiert (5 Node-Labels, 4 Relationship-Types, MERGE-Strategien) → `docs/status/kapitel5_implementierungsdetails.md` (Abschnitt III)
- **Ablation-Modi:** Alle Ablation-Flags dokumentiert (`use_claim_extraction`, `use_claim_verification`, `use_spans`, `ablation_mode`), Implementierung via Strategy-Pattern erklärt → `docs/status/kapitel5_implementierungsdetails.md` (Abschnitt IV)

### V.2 `docs/milestones/M6_claim_basierter_factuality_agent_und_evaluationsinfrastruktur.md`

**Überschrift:** Nach Abschnitt "2. Claim-Verifikation" ergänzen:

**Neuer Abschnitt: "2.1 Prompt-Details"**

**Bulletpoints:**
- **Claim-Extraction Prompt:** Extrahiert atomare, faktische Claims aus Sätzen, erzwingt Substring-Constraint (Claim muss wörtlich im Satz vorkommen), max 5 Claims → `app/services/agents/factuality/claim_extractor.py:163-205`
- **Claim-Verification Prompt:** Verifiziert Claims gegen Evidence-Passagen (wenn `evidence_context_list` vorhanden) oder Kontext (Fallback), erzwingt Evidence-Auswahl (`selected_evidence_index`, `evidence_quote`) → `app/services/agents/factuality/claim_verifier.py:448-540`
- **Evidence-Retrieval:** Nicht promptbasiert, sondern deterministisch via Sliding-Window + Jaccard-Similarity + Boosting für Zahlen/Entities → `app/services/agents/factuality/evidence_retriever.py:45-98`

### V.3 `docs/milestones/M9_explainability_modul.md`

**Überschrift:** Nach Abschnitt "2) ExplainabilityService" ergänzen:

**Neuer Abschnitt: "2.1 LLM-Parameter"**

**Bulletpoints:**
- **Zentrale Defaults:** Alle LLM-Calls verwenden `temperature=0.0` (deterministisch) und `max_tokens=800` (ausreichend für JSON-Responses) → `app/llm/openai_client.py:21-22`
- **Keine Agent-Overrides:** Factuality, Coherence und Readability Agenten verwenden die zentralen Defaults, keine Agent-spezifischen Parameter → `app/llm/openai_client.py:17-25`
- **LLM-as-a-Judge:** `temperature` kann via `judge_temperature` Parameter überschrieben werden (Default: 0.0) → `app/services/judges/llm_judge.py:40-47`, `app/services/verification_service.py:82-83`

### V.4 `docs/status/architektur_komplett.md`

**Überschrift:** Nach Abschnitt "D.2 Neo4j (Graph)" ergänzen:

**Neuer Abschnitt: "D.2.1 Graphschema (aus Code abgeleitet)"**

**Bulletpoints:**
- **Node-Labels:** Article, Summary, Run, Metric, IssueSpan (Dual-Label: `IssueSpan:Error`) → `app/db/neo4j/graph_persistence.py:64-134`
- **Relationship-Types:** `HAS_SUMMARY` (Article → Summary), `EVALUATES` (Run → Summary), `HAS_METRIC` (Summary → Metric), `HAS_ISSUE_SPAN` (Metric → IssueSpan) → `app/db/neo4j/graph_persistence.py:66-124`
- **MERGE-Strategie:** Nodes werden via `MERGE` erstellt (idempotent, verhindert Duplikate), Properties via `SET` aktualisiert → `app/db/neo4j/graph_persistence.py:64-134`
- **Key Properties:** Article/Summary/Run: `id` (MERGE-Key), Metric: `run_id` + `summary_id` + `dimension` (kombiniert), IssueSpan: `run_id` + `summary_id` + `dimension` + `span_index` (kombiniert) → `app/db/neo4j/graph_persistence.py:64-134`

---

## Zusammenfassung

Alle fehlenden Implementierungsdetails wurden aus dem Repository extrahiert:

1. ✅ **Prompts:** 11 Prompt-Texte dokumentiert (Claim-Extraction, Claim-Verification, Coherence, Readability, LLM-as-a-Judge), inkl. Versionierung und Output-Formate
2. ✅ **LLM-Parameter:** Zentrale Defaults dokumentiert (`temperature=0.0`, `max_tokens=800`), Priorität zwischen Client-Defaults und Agent-Overrides geklärt
3. ✅ **Neo4j Schema:** Vollständiges Graphschema dokumentiert (5 Node-Labels, 4 Relationship-Types, MERGE-Strategien)
4. ✅ **Ablation-Modi:** 7 Ablation-Flags dokumentiert, Implementierung via Strategy-Pattern erklärt

**Nicht gefunden (Suchspuren):**
- **Run-Tag "v3_uncertain_spans":** Wird in `configs/m10_factuality_runs.yaml` als `run_tag` verwendet (früher fälschlicherweise als `prompt_version`). Dient als Cache-/Run-ID-Tag für M10-Evaluationsläufe, ist nicht an Agent-Prompt-Versionierung gekoppelt. **Suchspur:** `grep -r "v3_uncertain_spans" app/` → keine Treffer (bestätigt: kein Agent-Prompt).
- **Seed-Parameter auf Client-Ebene:** `llm_seed: 42` in YAML-Configs vorhanden, aber nicht direkt an OpenAIClient übergeben (möglicherweise via Run-Manager). **Suchspur:** `grep -r "seed" app/llm/` → keine Treffer.

