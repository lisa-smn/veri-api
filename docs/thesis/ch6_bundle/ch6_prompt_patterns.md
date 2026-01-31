# Prompt Patterns im veri-api

**Erstellt:** 2026-01-30  
**Zweck:** Systematische Analyse der verwendeten Prompt-Strategien nach White et al. (2023)

---

## Tabelle: Prompt Patterns im veri-api

| Systemkomponente | Prompt-Pattern (White et al. 2023) | Prompt-Beleg (wortgleicher Auszug) | Zweck im System | Risiko / Limitierung | Fundstelle im Code |
|---|---|---|---|---|---|
| **FactualityAgent: Claim Extraction** | Template Pattern + Fact Check List Pattern | "Extrahiere NUR überprüfbare, FAKTISCHE Behauptungen (Claims). Ein Claim ist nur dann gültig, wenn: er konkret und überprüfbar ist (gegen einen Artikeltext), er KEINE reine Meinung/Bewertung ist, er NICHT über den Text selbst spricht (keine Meta-Aussagen über Lesbarkeit/Stil/Struktur). Gib NUR JSON im folgenden Format zurück: `{\"claims\": [{\"text\": \"Claim 1 (exakt aus dem Satz kopiert)\"}]}`" | Atomare Claims aus Sätzen extrahieren, Filterung von Meta-/Opinion-Statements, Substring-Constraint für Rückverfolgbarkeit | Format-Drift (LLM generiert nicht-JSON), Overconstraint (zu strikte Filter reduzieren Recall), Substring-Matching kann bei Paraphrasen fehlschlagen | `app/services/agents/factuality/claim_extractor.py:163-205` (`_build_prompt`) |
| **FactualityAgent: Claim Verification** | Template Pattern + Cognitive Verifier Pattern + Context Manager Pattern | "Verwende ausschließlich die EVIDENCE-PASSAGEN. Keine Weltkenntnis, keine Vermutungen. 'correct' NUR, wenn du eine Passage findest, die den Claim klar stützt. evidence_quote MUSS ein exakter Substring der ausgewählten Passage [selected_evidence_index] sein." | Verifikation gegen Evidence-Passagen mit Evidence-Gate (hard constraint: correct/incorrect nur mit evidence_found=True), wörtliche Quote-Validierung | False sense of correctness (LLM kann Evidence falsch interpretieren), Schema-Violations (selected_evidence_index/evidence_quote Inkonsistenzen), Coverage-Check kann zu konservativ sein | `app/services/agents/factuality/claim_verifier.py:448-502` (`_build_prompt`) |
| **FactualityAgent: Evidence Gate** | Cognitive Verifier Pattern (Code-Logik) | Gate-Logik: `label in {"correct","incorrect"}` nur erlaubt wenn `evidence_found == True`; `label == "incorrect"` ohne Evidence → `"uncertain"`; Coverage-Check validiert ob evidence_quote den Claim abdeckt (min_soft_token_coverage) | Reduziert False Positives durch harte Evidence-Anforderung, erzwingt wörtliche Zitate für Verifikation | Overconstraint (legitime Verifikationen ohne wörtliche Quote werden zu "uncertain"), Coverage-Check kann zu strikt sein | `app/services/agents/factuality/claim_verifier.py:319-365` (`_apply_gate`), `304-318` (`_coverage_check`) |
| **CoherenceAgent** | Template Pattern + Persona Pattern (implizit) | "Du bewertest NUR die KOHÄRENZ (Coherence) der SUMMARY. Bewerte NICHT: Lesbarkeit (Satzlänge, Kommas, Stil, Grammatik), Tonalität, faktische Korrektheit gegenüber dem Artikel. Gib NUR JSON zurück, ohne Text außerhalb des JSON." | Bewertung logischer Konsistenz, Informationsfluss, Redundanz; strikte Abgrenzung zu anderen Dimensionen | Format-Drift, Dimension-Overlap (LLM kann Lesbarkeit/Coherence verwechseln), Span-Mapping kann bei Paraphrasen fehlschlagen | `app/services/agents/coherence/coherence_verifier.py:117-160` (`_build_prompt`) |
| **ReadabilityAgent** | Template Pattern + Persona Pattern (implizit) | "Du bewertest NUR die LESBARKEIT (Readability) einer Summary. Bewerte KEINE: faktische Korrektheit, logische Kohärenz, Stil, Tonalität, 'Schönheit' der Sprache. Gib NUR JSON zurück, ohne zusätzliche Erklärungen außerhalb des JSON." | Bewertung Lesefluss, Satzlänge, Verschachtelung, Interpunktion; strikte Abgrenzung zu anderen Dimensionen | Format-Drift, Dimension-Overlap, Span-Mapping kann bei Paraphrasen fehlschlagen | `app/services/agents/readability/readability_verifier.py:159-206` (`_build_prompt_v1`), `208-247` (`_build_prompt_v2`) |
| **LLM-as-a-Judge (Readability/Coherence/Factuality)** | Template Pattern + Persona Pattern | "Du bewertest die LESBARKEIT/KOHÄRENZ/FAKTENTREUE einer Textzusammenfassung. WICHTIG: Nutze die VOLLE Skala (1-5). Nicht nur 4-5 verwenden! Output-Schema (MUSS exakt eingehalten werden, NUR JSON): `{\"rating\": 3, \"confidence\": 0.8, \"rationale\": \"...\"}`" | Direkte Bewertung ohne Agent-Pipeline, Vergleichsbasis für Agent-Performance, Skala-Anker zur Vermeidung von Collapse | Scale Collapse (LLM nutzt nur Teil der Skala trotz "use full scale"), Format-Drift, Confidence kann nicht kalibriert werden | `app/services/judges/prompts.py:26-56` (`_build_readability_prompt_v1`), `116-145` (`_build_coherence_prompt_v1`), `166-196` (`_build_factuality_prompt_v1`) |
| **Robustheitsmechanismen (Parsing/Validation)** | Custom Pattern (kein direkter Match) | Post-Processing: JSON-Extraktion via `raw.find("{")` / `raw.rfind("}")`, Schema-Validierung via Pydantic (`VerifierLLMOutput.model_validate`), Fallback bei Parse-Fehler (→ "uncertain"), Strict-Mode für fail-fast bei Schema-Violations | Fehlerbehandlung bei Format-Drift, Validierung von Evidence-Quote als Substring, Coverage-Check als zusätzliche Validierung | Parse-Fehler können zu Information Loss führen (nur JSON-Extraktion, Rest wird verworfen), Strict-Mode kann zu aggressiv sein (fail-fast bei kleinen Schema-Abweichungen) | `app/services/agents/factuality/claim_verifier.py:149-186` (`_parse_llm_output`), `app/services/agents/coherence/coherence_verifier.py:162-180` (`_parse_output`) |

---

## Einführung und Interpretation

Prompt Patterns nach White et al. (2023) sind wiederverwendbare Bausteine für die Konstruktion effektiver LLM-Prompts. Im veri-api-System werden mehrere dieser Patterns kombiniert, um eine robuste, erklärbare Evaluationspipeline zu realisieren. Die konzeptionelle Zuordnung zu White et al. (2023) dient der strukturierten Analyse; nicht alle Implementierungen entsprechen exakt den Original-Patterns, da system-spezifische Anforderungen (z.B. Evidence-Gate, Substring-Constraints) zusätzliche Constraints einführen.

Das **Template Pattern** dominiert alle Prompts durch strikte JSON-Schema-Vorgaben, was die maschinelle Verarbeitbarkeit sichert, aber Format-Drift-Risiken birgt. Das **Cognitive Verifier Pattern** manifestiert sich im Evidence-Gate der Claim-Verifikation, wo "correct"/"incorrect" nur mit belastbarer Evidence (wörtliche Zitate) erlaubt sind. Das **Context Manager Pattern** ist in allen Factuality-Prompts präsent ("Verwende ausschließlich die EVIDENCE-PASSAGEN. Keine Weltkenntnis"), um Halluzinationen zu reduzieren. Das **Fact Check List Pattern** strukturiert die Claim-Extraktion durch explizite Kriterien (was ist ein Claim, was nicht).

**Limitierungen:** Die strikten Schema-Constraints können zu Overconstraint führen (z.B. Evidence-Gate verwirft legitime Verifikationen ohne wörtliche Quote). Format-Drift bleibt ein Risiko trotz Post-Processing (JSON-Extraktion). Dimension-Overlap kann auftreten, wenn LLMs trotz expliziter Abgrenzung Lesbarkeit/Coherence/Factuality verwechseln. Die "use full scale"-Anweisung in Judge-Prompts kann Scale Collapse nicht vollständig verhindern.

---

## Fundstellen-Liste (zur Nachprüfbarkeit)

### Claim Extraction
- `app/services/agents/factuality/claim_extractor.py:163-205` (`_build_prompt`)

### Claim Verification
- `app/services/agents/factuality/claim_verifier.py:448-502` (`_build_prompt` mit evidence_context_list)
- `app/services/agents/factuality/claim_verifier.py:503-542` (`_build_prompt` ohne evidence_context_list, Fallback)

### Evidence Gate (Code-Logik)
- `app/services/agents/factuality/claim_verifier.py:319-365` (`_apply_gate`)
- `app/services/agents/factuality/claim_verifier.py:304-318` (`_coverage_check`)

### Coherence Agent
- `app/services/agents/coherence/coherence_verifier.py:117-160` (`_build_prompt`)

### Readability Agent
- `app/services/agents/readability/readability_verifier.py:159-206` (`_build_prompt_v1`)
- `app/services/agents/readability/readability_verifier.py:208-247` (`_build_prompt_v2`)

### LLM-as-a-Judge
- `app/services/judges/prompts.py:26-56` (`_build_readability_prompt_v1`)
- `app/services/judges/prompts.py:58-97` (`_build_readability_prompt_v2_float`)
- `app/services/judges/prompts.py:116-145` (`_build_coherence_prompt_v1`)
- `app/services/judges/prompts.py:166-196` (`_build_factuality_prompt_v1`)
- `app/services/judges/prompts.py:199-239` (`_build_factuality_prompt_v2_binary`)

### Robustheitsmechanismen
- `app/services/agents/factuality/claim_verifier.py:149-186` (`_parse_llm_output`)
- `app/services/agents/coherence/coherence_verifier.py:162-180` (`_parse_output`)
- `app/services/agents/readability/readability_verifier.py:353-380` (`_parse_output`)

---

## Literaturverweis

White, J., Fu, Q., Hays, S., Sandborn, M., Olea, C., Gilbert, H., Elnashar, A., Spencer-Smith, J., & Schmidt, D. C. (2023). A prompt pattern catalog to enhance prompt engineering with ChatGPT. *arXiv preprint arXiv:2302.11382*.
