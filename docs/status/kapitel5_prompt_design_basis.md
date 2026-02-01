# Prompt Design – Basis für Kapitel 5.6

**Datum:** 2026-01-27  
**Zweck:** Systematische Analyse aller Prompt-Templates im Code für Unterkapitel 5.6 "Prompt Design"

---

## A) Prompt-Katalog

| Component | File:Line | Versions | Inputs | Output Schema | Key Constraints |
|-----------|----------|----------|--------|---------------|-----------------|
| **Claim Extraction** | `app/services/agents/factuality/claim_extractor.py:163-205` | v1 (fest) | `sentence: str` | `{"claims": [{"text": str}]}` | Max 5 claims, `text` muss Substring von `sentence` sein, min 4 Zeichen |
| **Claim Verification** | `app/services/agents/factuality/claim_verifier.py:448-502` | v1 (fest) | `evidence_context_list: list[str]` (oder `context: str`), `claim_text: str` | `{"label": "correct\|incorrect\|uncertain", "confidence": float, "error_type": "ENTITY\|NUMBER\|DATE\|OTHER"\|null, "explanation": str, "selected_evidence_index": int, "evidence_quote": str\|null}` | `evidence_quote` muss Substring der Passage `[selected_evidence_index]` sein (wenn index >= 0), `selected_evidence_index=-1` => `evidence_quote=null` |
| **Coherence (Agent)** | `app/services/agents/coherence/coherence_verifier.py:117-160` | v1 (fest) | `article_text: str`, `summary_text: str` | `{"score": float [0,1], "explanation": str, "issues": [{"type": "LOGICAL_INCONSISTENCY\|CONTRADICTION\|REDUNDANCY\|ORDERING\|OTHER", "severity": "low\|medium\|high", "summary_span": str, "comment": str, "hint": str\|null}]}` | Max 8 issues, `summary_span` muss Substring von `summary_text` sein, wenn `score < 0.7` dann min 1 issue |
| **Readability (Agent) v1** | `app/services/agents/readability/readability_verifier.py:159-206` | v1 | `article_text: str`, `summary_text: str` | `{"score": float [0,1], "explanation": str, "issues": [{"type": "LONG_SENTENCE\|COMPLEX_NESTING\|PUNCTUATION_OVERLOAD\|HARD_TO_PARSE", "severity": "low\|medium\|high", "summary_span": str, "comment": str, "metric": str\|null, "metric_value": float\|null}]}` | Max 8 issues, `summary_span` muss Substring von `summary_text` sein, wenn `score < 0.7` dann min 1 issue |
| **Readability (Agent) v2** | `app/services/agents/readability/readability_verifier.py:208-254` | v2 | `article_text: str`, `summary_text: str` | `{"score_raw_1_to_5": int [1-5], "rationale": str, "explanation": str, "issues": [...]}` | Integer 1-5 (nicht float), wird intern zu 0-1 normalisiert (`app/services/agents/readability/readability_verifier.py:353-365`) |
| **Judge Readability v1** | `app/services/judges/prompts.py:26-56` | v1 | `summary_text: str`, `article_text: str\|None` | `{"rating": int [1-5], "confidence": float [0,1], "rationale": str}` | Rating 1-5, wird zu 0-1 normalisiert (`app/services/judges/parsing.py:107-138`) |
| **Judge Readability v2_float** | `app/services/judges/prompts.py:58-97` | v2_float | `summary_text: str`, `article_text: str\|None` | `{"score": float [0.00-1.00], "confidence": float [0,1], "rationale": str}` | Score bereits 0-1, zwei Dezimalstellen |
| **Judge Coherence v1** | `app/services/judges/prompts.py:116-145` | v1 | `summary_text: str`, `article_text: str\|None` | `{"rating": int [1-5], "confidence": float [0,1], "rationale": str}` | Rating 1-5, wird zu 0-1 normalisiert |
| **Judge Factuality v1** | `app/services/judges/prompts.py:166-196` | v1 | `summary_text: str`, `article_text: str` | `{"rating": int [1-5], "confidence": float [0,1], "rationale": str}` | Rating 1-5, wird zu 0-1 normalisiert |
| **Judge Factuality v2_binary** | `app/services/judges/prompts.py:199-239` | v2_binary | `summary_text: str`, `article_text: str` | `{"error_present": bool, "confidence": float [0,1], "rationale": str}` | Binary verdict, `error_present=true` → score_norm=0.0, `false` → 1.0 (`app/services/judges/llm_judge.py:208-212`) |

---

## B) Designprinzipien

1. **Strukturierter JSON-Output:** Alle Prompts erzwingen striktes JSON-Output mit festem Schema (`app/services/judges/prompts.py:4`). JSON wird durch Prompt-Constraints erzwungen, nicht durch `response_format`-Parameter (`app/llm/openai_client.py:17-25` verwendet keine `response_format`).

2. **Substring-Constraints für Traceability:** Claims (`app/services/agents/factuality/claim_extractor.py:237-241`), Evidence-Quotes (`app/services/agents/factuality/claim_verifier.py:276-290`), Summary-Spans (`app/services/agents/coherence/coherence_verifier.py:187-197`, `app/services/agents/readability/readability_verifier.py:363-375`) müssen als wörtliche Substrings im Originaltext vorkommen, um Char-Positionen zu mappen.

3. **Robustes JSON-Parsing:** Alle Parser verwenden das Pattern "find first `{` ... last `}`" (`app/services/agents/factuality/claim_extractor.py:209-211`, `app/services/agents/factuality/claim_verifier.py:158-163`, `app/services/agents/coherence/coherence_verifier.py:164-167`, `app/services/judges/parsing.py:32-36`), um JSON aus Markdown-umgebenen Outputs zu extrahieren.

4. **Fallback-Strategien bei Parse-Fehlern:** Claim-Extraction gibt leere Liste zurück (`app/services/agents/factuality/claim_extractor.py:212-213`), Coherence/Readability verwenden Default-Score 0.0 und leere Issue-Listen (`app/services/agents/coherence/coherence_verifier.py:168-173`), Claim-Verification verwendet `uncertain` mit Confidence 0.0 (`app/services/agents/factuality/claim_verifier.py:179-186`), Judge verwendet Retry-Logik mit Regex-Fallback (`app/services/judges/parsing.py:41-50`, `app/services/judges/llm_judge.py:197-201`).

5. **Evidence-gebundene Labels (Gate):** Claim-Verification erfordert, dass `evidence_quote` ein Substring der ausgewählten Passage ist (`app/services/agents/factuality/claim_verifier.py:276-290`). Das Evidence-Gate erlaubt "incorrect"-Labels nur bei `evidence_found=True` (`app/services/agents/factuality/claim_verifier.py:200-250`).

6. **Span-Validierung:** Alle Spans (`summary_span`, `evidence_quote`) werden auf Substring-Eigenschaft geprüft. Bei Fehlschlag werden Fallbacks verwendet: Coherence/Readability verwenden ersten 80-120 Zeichen der Summary (`app/services/agents/coherence/coherence_verifier.py:194-195`, `app/services/agents/readability/readability_verifier.py:375-380`).

7. **Prompt-Versionierung:** Readability-Agent unterstützt `v1`/`v2` (`app/services/agents/readability/readability_agent.py:36-40`), Judge-Prompts unterstützen `v1`/`v2_float` (Readability), `v1` (Coherence), `v1`/`v2_binary` (Factuality) (`app/services/judges/prompts.py:8-240`). Coherence-Agent und Factuality-Agent (Claim-Extraction/Verification) verwenden feste Prompts ohne Versionierung.

8. **Trennung run_tag vs. prompt_version:** `run_tag` wird für Cache-Identifikation und Run-Labeling verwendet (`scripts/run_m10_factuality.py:200-228`), während `prompt_version` nur für echte Prompt-Versionierung existiert. Legacy-Mapping erlaubt `prompt_version` als Fallback für `run_tag`, mit Deprecation-Warnung.

9. **LLM-Defaults zentral:** `OpenAIClient` verwendet `temperature=0.0` (deterministisch), `max_tokens=800` (ausreichend für JSON-Responses) (`app/llm/openai_client.py:21-22`). Kein `seed`-Parameter wird übergeben; Determinismus wird über `temperature=0.0` erreicht.

10. **Strict-Mode für Claim-Verification:** `LLMClaimVerifier` unterstützt `strict_mode` (`app/services/agents/factuality/claim_verifier.py:172-176`), der bei Parse-Fehlern einen `ValueError` wirft statt Fallback zu verwenden.

11. **Normalisierung für robustes Matching:** Quote-Matching verwendet Unicode-Normalisierung (NFKC), Whitespace-Normalisierung und Quote-Normalisierung (`app/services/agents/factuality/claim_verifier.py:188-218`, `app/services/agents/factuality/claim_extractor.py:280-327`), um kleine Abweichungen zu tolerieren.

12. **Max-Constraints:** Claim-Extraction: max 5 claims (`app/services/agents/factuality/claim_extractor.py:198`), Coherence/Readability: max 8 issues (`app/services/agents/coherence/coherence_verifier.py:37`, `app/services/agents/readability/readability_verifier.py:197`), um Output-Größe zu begrenzen.

---

## C) Parsing-, Validierungs- und Fallback-Logik

### JSON-Parsing-Strategien

**Pattern:** Alle Parser verwenden `start = raw.find("{")` und `end = raw.rfind("}") + 1`, um JSON aus Markdown-umgebenen Outputs zu extrahieren:
- Claim-Extraction: `app/services/agents/factuality/claim_extractor.py:209-211`
- Claim-Verification: `app/services/agents/factuality/claim_verifier.py:158-163`
- Coherence: `app/services/agents/coherence/coherence_verifier.py:164-167`
- Readability: `app/services/agents/readability/readability_verifier.py:291-294`
- Judge: `app/services/judges/parsing.py:32-36`

**Fallback:** Judge verwendet zusätzlich Regex-Extraktion (`app/services/judges/parsing.py:41-50`) für `rating`, `score`, `error_present`, `rationale`, `confidence`.

### Validierungen

**Substring-Constraints:**
- Claims: `_normalize_to_sentence_substring()` prüft, ob Claim-Text ein Substring des Satzes ist (`app/services/agents/factuality/claim_extractor.py:237-241`, `280-327`). Bei Fehlschlag wird Claim verworfen.
- Evidence-Quotes: `_validate_evidence()` prüft, ob `evidence_quote` in Passage `[selected_evidence_index]` vorkommt (`app/services/agents/factuality/claim_verifier.py:276-290`). Bei Fehlschlag wird `evidence_found=False` gesetzt.
- Summary-Spans: `_ensure_spans_are_substrings()` prüft, ob `summary_span` in `summary_text` vorkommt (`app/services/agents/coherence/coherence_verifier.py:187-197`, `app/services/agents/readability/readability_verifier.py:363-375`). Bei Fehlschlag wird Fallback-Span (erste 80-120 Zeichen) verwendet.

**Schema-Validierung:**
- Claim-Verification verwendet Pydantic-Modell `VerifierLLMOutput` (`app/services/agents/factuality/claim_verifier.py:166`).
- Judge verwendet `expected_schema`-Dict für Typ-Checks (`app/services/judges/llm_judge.py:188-194`).

### Fallback-Strategien

**Bei Parse-Fehlern:**
- Claim-Extraction: leere Liste (`app/services/agents/factuality/claim_extractor.py:212-213`)
- Claim-Verification: `uncertain` mit Confidence 0.0 (`app/services/agents/factuality/claim_verifier.py:179-186`), außer in Strict-Mode (wirft `ValueError`)
- Coherence: `score=0.0`, leere Issues, Fallback-Explanation (`app/services/agents/coherence/coherence_verifier.py:168-173`)
- Readability: ähnlich wie Coherence
- Judge: Retry-Logik (max `retries` Versuche, `app/services/judges/llm_judge.py:179-201`), dann Regex-Fallback (`app/services/judges/parsing.py:41-50`)

**Bei Validierungs-Fehlern:**
- Claims ohne Substring: verworfen (`app/services/agents/factuality/claim_extractor.py:238-241`)
- Evidence-Quotes ohne Substring: `evidence_found=False` (`app/services/agents/factuality/claim_verifier.py:283-290`)
- Spans ohne Substring: Fallback-Span (erste 80-120 Zeichen, `app/services/agents/coherence/coherence_verifier.py:194-195`)

---

## D) LLM-Call-Konfiguration

**Zentrale Defaults (`app/llm/openai_client.py:17-25`):**
- `temperature=0.0` (deterministisch)
- `max_tokens=800` (ausreichend für JSON-Responses)
- `model`: wird bei Client-Initialisierung übergeben (standardmäßig `gpt-4o-mini`)

**Keine explizite JSON-Mode-Konfiguration:**
- Kein `response_format="json_object"` oder `json_schema`-Parameter
- JSON wird ausschließlich durch Prompt-Constraints erzwungen

**Kein Seed-Parameter:**
- `llm_seed=42` wird in Configs dokumentiert (`configs/m10_factuality_runs.yaml:14`), aber nicht an `chat.completions.create()` übergeben
- Determinismus wird über `temperature=0.0` erreicht

**Judge-spezifische Parameter:**
- `LLMJudge` unterstützt `default_temperature`, `default_n` (Committee-Judgements), `default_aggregation` (`app/services/judges/llm_judge.py:34-48`)
- Standard: `temperature=0.0`, `n=1`, `aggregation="mean"`

---

## E) Versionierung und Reproduzierbarkeit

### Echte Prompt-Versionierung

**Readability-Agent:**
- `v1`: Score 0-1, Issues mit optionalen Metriken (`app/services/agents/readability/readability_verifier.py:159-206`)
- `v2`: Rubrik-basiert, Score 1-5 (Integer), wird intern zu 0-1 normalisiert (`app/services/agents/readability/readability_verifier.py:208-254`, `353-365`)
- Version wird über `prompt_version`-Parameter gesteuert (`app/services/agents/readability/readability_agent.py:36-40`)

**Judge-Prompts:**
- Readability: `v1` (1-5 Rating), `v2_float` (0.00-1.00 Score) (`app/services/judges/prompts.py:8-97`)
- Coherence: `v1` (1-5 Rating) (`app/services/judges/prompts.py:100-145`)
- Factuality: `v1` (1-5 Rating), `v2_binary` (binary `error_present`) (`app/services/judges/prompts.py:148-239`)
- Version wird über `prompt_version`-Parameter gesteuert (`app/services/judges/llm_judge.py:57-72`)

**Keine Versionierung:**
- Claim-Extraction: fester Prompt (`app/services/agents/factuality/claim_extractor.py:163-205`)
- Claim-Verification: fester Prompt (`app/services/agents/factuality/claim_verifier.py:448-502`)
- Coherence-Agent: fester Prompt (`app/services/agents/coherence/coherence_verifier.py:117-160`)

### Run-Tag vs. Prompt-Version

**Trennung:**
- `run_tag`: für Cache-Identifikation und Run-Labeling (`scripts/run_m10_factuality.py:200-228`)
- `prompt_version`: nur für echte Prompt-Versionierung (wenn verschiedene Prompt-Varianten existieren)

**Legacy-Mapping:**
- `run_tag = run_config.get("run_tag") or run_config.get("prompt_version") or "default"` (`scripts/run_m10_factuality.py:202-207`)
- Deprecation-Warnung, wenn `prompt_version` als `run_tag` verwendet wird (`scripts/run_m10_factuality.py:210-213`)
- Fail-fast bei Ambiguität (beide gesetzt und unterschiedlich, `scripts/run_m10_factuality.py:216-221`)

**Cache-Key:**
- Basiert auf Artikel-Text, Summary-Text, Modell und `run_tag` (`scripts/run_m10_factuality.py:154-170`)
- `run_tag` ist Teil des Cache-Dateinamens (`..._{run_tag}.jsonl`)

---

## F) Offene Punkte / Unsicherheiten

1. **Seed-Parameter:** `llm_seed=42` wird in Configs dokumentiert (`configs/m10_factuality_runs.yaml:14`), aber nicht an `OpenAIClient.complete()` übergeben. Gesucht wurde in `app/llm/openai_client.py` und `app/llm/llm_client.py`. Möglicherweise wird Determinismus ausschließlich über `temperature=0.0` erreicht.

2. **JSON-Mode (response_format):** Es wurde kein expliziter `response_format="json_object"` oder `json_schema`-Parameter gefunden. Gesucht wurde nach `response_format`, `json_mode`, `tool_call` in `app/`. JSON wird ausschließlich durch Prompt-Constraints erzwungen.

3. **Repair-Prompts für Agenten:** Während der LLM-Judge Repair-Prompts bei fehlerhaftem JSON verwendet (`app/services/judges/llm_judge.py:163-201`), verwenden die Agenten (Factuality, Coherence, Readability) keine expliziten Repair-Prompts. Bei Parsing-Fehlern werden Fallbacks verwendet (leere Listen, Default-Scores).

4. **Strict-Mode Verwendung:** `LLMClaimVerifier` unterstützt `strict_mode` (`app/services/agents/factuality/claim_verifier.py:172-176`), aber es ist unklar, wo dieser Modus aktiviert wird. Gesucht wurde in `app/services/agents/factuality/factuality_agent.py:92-97` (wird über `strict_mode`-Parameter gesteuert).

5. **Evidence-Coverage-Check:** `_evidence_covers_claim()` (`app/services/agents/factuality/claim_verifier.py:544-617`) prüft, ob Evidence den Claim abdeckt (z.B. Zahlen/Entitäten müssen im Evidence vorkommen). Es ist unklar, ob dieser Check Teil des Gate-Prozesses ist oder nur für Confidence-Anpassungen verwendet wird.

---

## G) Repo-Belegliste (Aussage → Datei:Zeilen)

- **Claim-Extraction Prompt:** `app/services/agents/factuality/claim_extractor.py:163-205` (Prompt-Text), `app/services/agents/factuality/claim_extractor.py:207-258` (Parsing + Substring-Validierung)
- **Claim-Verification Prompt:** `app/services/agents/factuality/claim_verifier.py:448-502` (Prompt-Text), `app/services/agents/factuality/claim_verifier.py:149-187` (Parsing), `app/services/agents/factuality/claim_verifier.py:220-290` (Evidence-Validierung)
- **Coherence Prompt:** `app/services/agents/coherence/coherence_verifier.py:117-160` (Prompt-Text), `app/services/agents/coherence/coherence_verifier.py:162-184` (Parsing), `app/services/agents/coherence/coherence_verifier.py:187-197` (Span-Validierung)
- **Readability Prompts:** `app/services/agents/readability/readability_verifier.py:159-206` (v1), `app/services/agents/readability/readability_verifier.py:208-254` (v2), `app/services/agents/readability/readability_verifier.py:291-365` (Parsing + Normalisierung)
- **Judge Prompts:** `app/services/judges/prompts.py:8-240` (alle Dimensionen und Versionen)
- **JSON-Parsing Pattern:** `app/services/agents/factuality/claim_extractor.py:209-211`, `app/services/agents/factuality/claim_verifier.py:158-163`, `app/services/agents/coherence/coherence_verifier.py:164-167`, `app/services/judges/parsing.py:32-36`
- **Substring-Constraints:** `app/services/agents/factuality/claim_extractor.py:237-241, 280-327`, `app/services/agents/factuality/claim_verifier.py:276-290`, `app/services/agents/coherence/coherence_verifier.py:187-197`, `app/services/agents/readability/readability_verifier.py:363-375`
- **Fallback-Strategien:** `app/services/agents/factuality/claim_extractor.py:212-213`, `app/services/agents/factuality/claim_verifier.py:179-186`, `app/services/agents/coherence/coherence_verifier.py:168-173`, `app/services/judges/llm_judge.py:197-201`
- **LLM Defaults:** `app/llm/openai_client.py:17-25` (temperature=0.0, max_tokens=800)
- **Run-Tag vs Prompt-Version:** `scripts/run_m10_factuality.py:200-228` (Legacy-Mapping), `configs/m10_factuality_runs.yaml:15` (run_tag), `scripts/aggregate_factuality_runs.py:159-166` (Extraction)
- **Evidence-Gate:** `app/services/agents/factuality/claim_verifier.py:200-250` (Gate-Logik), `tests/unit/test_evidence_gate_refactored.py` (Tests)

