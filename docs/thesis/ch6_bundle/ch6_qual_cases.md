# Chapter 6 – Qualitative Casebook

**Quelle:** `results/evaluation/runs/results/evidence_gate_test_count_as_error_examples.jsonl`  
**Run:** `evidence_gate_test_count_as_error`  
**Dataset:** FRANK (n=50)

---

## Factuality Cases (FRANK)

### TP-1: Evidence vorhanden (ex_0)

**JSONL Zeile:** 1  
**Artefaktpfad:** `results/evaluation/runs/results/evidence_gate_test_count_as_error_examples.jsonl` (Zeile 1)

**Labels:**
- Gold (has_error): `true`
- Prediction (has_error): `true`
- Score: `0.0`

**Summary:**
```
the fbi has said it will help the san bernardino killer to access iphones used by the san bernardino victims.
```

**Issue Span:**
- `issue_type`: `OTHER`
- `verdict`: `incorrect`
- `evidence_found`: `true`
- `severity`: `low`
- `confidence`: `0.5`
- `message`: "Satz 1: Claim 'the fbi has said it will help the san bernardino killer to access iphones' – Die Passage beschreibt, dass die FBI in einem anderen Fall in Arkansas helfen wird, nicht dass sie dem San Bernardino-Killer helfen werden, auf iPhones zuzugreifen."

**Evidence Quote:**
- `evidence_quote`: `null` (nicht in aktueller Examples-Datei gespeichert)
- **Hinweis:** Nach Re-Run mit aktualisiertem Code (vgl. `feat/store-evidence-quotes` Branch) wird `issue_spans[0].evidence_quote` die wörtliche Passage aus dem Artikel enthalten.

**Meta:**
- `hash`: `35933239`
- `model_name`: `BERTS2S`
- `factuality`: `0.0`

---

### TP-2: ENTITY Error (ex_4)

**JSONL Zeile:** 5  
**Artefaktpfad:** `results/evaluation/runs/results/evidence_gate_test_count_as_error_examples.jsonl` (Zeile 5)

**Labels:**
- Gold (has_error): `true`
- Prediction (has_error): `true`
- Score: `0.0`

**Summary:**
```
sweden's foreign minister johan gustsson has been freed after being kidnapped by islamist militants in mali.
```

**Issue Span:**
- `issue_type`: `ENTITY`
- `verdict`: `incorrect`
- `evidence_found`: `true`
- `severity`: `medium`
- `confidence`: `0.5`
- `message`: "Satz 1: Claim 'sweden's foreign minister johan gustsson has been freed after being kidnapped by islamist militants in mali.' – Der Claim verwechselt den Namen; Johan Gustafsson ist nicht der schwedische Außenminister, sondern ein entführter Schwede. Der Außenminister ist Margot Wallström."

**Evidence Quote:**
- `evidence_quote`: `null` (nicht in aktueller Examples-Datei gespeichert)
- **Hinweis:** Nach Re-Run wird `issue_spans[0].evidence_quote` die relevante Passage enthalten, die zeigt, dass Johan Gustafsson nicht der Außenminister ist.

**Meta:**
- `hash`: `40407620`
- `model_name`: `BERTS2S`
- `factuality`: `0.0`

---

### UNC-1: No Evidence (ex_13)

**JSONL Zeile:** 14  
**Artefaktpfad:** `results/evaluation/runs/results/evidence_gate_test_count_as_error_examples.jsonl` (Zeile 14)

**Labels:**
- Gold (has_error): `true`
- Prediction (has_error): `true` (wegen `uncertainty_policy="count_as_error"`)
- Score: `0.5`

**Summary:**
```
one of the uk's biggest commercial ships has been given the go-ahead by the ministry of defence (mod).
```

**Issue Span:**
- `issue_type`: `OTHER`
- `verdict`: `uncertain`
- `evidence_found`: `false`
- `severity`: `low`
- `confidence`: `0.5`
- `message`: "Satz 1: Claim 'by the ministry of defence (mod)' – Nicht sicher verifizierbar (Quelle zu vage/fehlend). Die Passage erwähnt das Ministerium für Verteidigung, aber es gibt keine spezifische Aussage oder Unterstützung für die Behauptung, die sich auf das MOD bezieht."

**Evidence Quote:**
- `evidence_quote`: `null` (keine Evidence gefunden)

**Meta:**
- `hash`: `21326309`
- `model_name`: `TConvS2S`
- `factuality`: `0.0`

---

### UNC-2: No Evidence (ex_15)

**JSONL Zeile:** 16  
**Artefaktpfad:** `results/evaluation/runs/results/evidence_gate_test_count_as_error_examples.jsonl` (Zeile 16)

**Labels:**
- Gold (has_error): `true`
- Prediction (has_error): `true` (wegen `uncertainty_policy="count_as_error"`)
- Score: `0.5`

**Summary:**
```
plans to build a new generation of royal navy frigates on the isle of wight have been submitted to the government.
```

**Issue Span:**
- `issue_type`: `OTHER`
- `verdict`: `uncertain`
- `evidence_found`: `false`
- `severity`: `low`
- `confidence`: `0.5`
- `message`: "Satz 1: Claim 'plans to build a new generation of royal navy frigates on the isle of wight have been submitted to the government' – Nicht sicher verifizierbar (Quelle zu vage/fehlend). Die bereitgestellten Passagen enthalten keine Informationen über Pläne zum Bau einer neuen Generation von Royal Navy-Fregatten auf der Isle of Wight."

**Evidence Quote:**
- `evidence_quote`: `null` (keine Evidence gefunden)

**Meta:**
- `hash`: `21326309`
- `model_name`: `TranS2S`
- `factuality`: `0.0`

---

### FN-1: Miss (ex_16)

**JSONL Zeile:** 17  
**Artefaktpfad:** `results/evaluation/runs/results/evidence_gate_test_count_as_error_examples.jsonl` (Zeile 17)

**Labels:**
- Gold (has_error): `true`
- Prediction (has_error): `false`
- Score: `0.5`

**Summary:**
```
australia's prime minister tony abbott has said his country has " no regrets " over the controversial immigration policy.
```

**Issue Spans:**
- `issue_spans`: `[]` (leer)
- **Grund:** Keine Issues gefunden, daher `pred_has_error=false`

**Meta:**
- `hash`: `34148931`
- `model_name`: `BERTS2S`
- `factuality`: `0.0`

---

### FN-2: Miss (ex_17)

**JSONL Zeile:** 18  
**Artefaktpfad:** `results/evaluation/runs/results/evidence_gate_test_count_as_error_examples.jsonl` (Zeile 18)

**Labels:**
- Gold (has_error): `true`
- Prediction (has_error): `false`
- Score: `0.5`

**Summary:**
```
australian prime minister tony abbott has called for a ban on migrants entering the mediterranean sea.
```

**Issue Spans:**
- `issue_spans`: `[]` (leer)
- **Grund:** Keine Issues gefunden, daher `pred_has_error=false`

**Meta:**
- `hash`: `34148931`
- `model_name`: `TConvS2S`
- `factuality`: `0.0`

---

## Coherence Cases (SummEval)

**Quelle:** `results/evaluation/coherence/coherence_20260107_205123_gpt-4o-mini_v1_seed42/predictions.jsonl`

### Coherence Example 1

**Example ID:** `dm-test-8764fb95bfad8ee849274873a92fb8d6b400eee2`  
**Artefaktpfad:** `results/evaluation/coherence/coherence_20260107_205123_gpt-4o-mini_v1_seed42/predictions.jsonl` (Zeile 1)

**Scores:**
- Ground Truth (raw): `1.333` (normalized: `0.083`)
- Prediction (agent): `0.3` (normalized, mapped to 1-5: `2.2`)

**Issues:**
- `num_issues`: `3`
- `max_severity`: `high`
- `issue_types_counts`: `{"ORDERING": 2, "REDUNDANCY": 1}`
- `top_issues`:
  - `{"type": "ORDERING", "severity": "high", "span_indices": [0, 99]}`
  - `{"type": "ORDERING", "severity": "high", "span_indices": [100, 168]}`
  - `{"type": "REDUNDANCY", "severity": "medium", "span_indices": [307, 356]}`

**Interpretation:**
- Agent erkennt niedrige Coherence (pred=0.3), was mit niedrigem GT (0.083) übereinstimmt.
- Hauptprobleme: ORDERING (2x high severity) und REDUNDANCY (1x medium).

---

## Readability Cases (SummEval)

**Quelle:** `results/evaluation/readability/readability_20260116_170832_gpt-4o-mini_v1_seed42/predictions.jsonl`

### Readability Example 1

**Example ID:** `dm-test-8764fb95bfad8ee849274873a92fb8d6b400eee2_7011036f`  
**Artefaktpfad:** `results/evaluation/readability/readability_20260116_170832_gpt-4o-mini_v1_seed42/predictions.jsonl` (Zeile 1)

**Scores:**
- Ground Truth (raw): `3.0` (normalized: `0.5`)
- Prediction (agent): `0.5` (normalized, mapped to 1-5: `3.0`)

**Issues:**
- `num_issues`: `2`
- `max_severity`: `medium`
- `top_issues`:
  - `{"type": null, "severity": "medium", "span_indices": [0, 99]}`
  - `{"type": null, "severity": "medium", "span_indices": [239, 306]}`

**Interpretation:**
- Agent erkennt mittlere Readability (pred=0.5), was mit GT (0.5) übereinstimmt.
- Zwei medium-severity Issues identifiziert.

---

## Notes

1. **Evidence Quotes:** Die aktuellen Factuality Examples enthalten noch keine `evidence_quote` Felder, da der Run vor der Code-Änderung erstellt wurde. Nach Re-Run mit aktualisiertem Code (Branch `feat/store-evidence-quotes`) werden `issue_spans[].evidence_quote` und `claims[].evidence_quote` verfügbar sein.

2. **Article/Summary Excerpts:** Für vollständige Artikeltexte siehe:
   - FRANK: `data/frank/frank_clean.jsonl` (suche nach `hash` in `meta`)
   - SummEval: `data/sumeval/sumeval_clean.jsonl` (suche nach `example_id`)

3. **Coherence/Readability:** Die `predictions.jsonl` enthalten keine vollständigen `article`/`summary` Felder, nur `example_id`. Für vollständige Texte siehe das entsprechende Dataset.
