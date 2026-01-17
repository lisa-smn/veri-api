# Readability Score Mechanism

**Datum:** 2026-01-15

## Entscheidung

**Mechanism:** LLM-judge (primär), mit Heuristik-Fallback für Issues

**Primary driver:** LLM rating mapped to [0,1] (v1: direkt, v2: 1-5 integer → normalisiert)

**Risk hotspot:** Prompt parsing / JSON extraction / Output collapse (v2 zeigt Kollaps auf ~0.25)

---

## Score Pipeline

1. **ReadabilityAgent.run()** → ruft `evaluator.evaluate()` (standard: `LLMReadabilityEvaluator`)
2. **LLMReadabilityEvaluator.evaluate()**:
   - Baut Prompt (v1 oder v2) mit `article_text` und `summary_text`
   - **LLM-Call:** `self.llm.complete(prompt)` → roher String-Output
   - **Parsing:** Extrahiert JSON aus LLM-Output (`_parse_output()`)
   - **Score-Extraktion:**
     - v1: `score = data.get("score", 0.0)` (direkt 0-1)
     - v2: `score_raw_1_to_5` (integer 1-5) → normalisiert: `(score_raw - 1) / 4`
   - **Issues:** Aus `data.get("issues", [])` geparst
   - **Heuristik-Fallback:** Wenn `score < 0.7` und keine Issues → `_heuristic_fallback_issues()`
3. **ReadabilityAgent**:
   - Clamp Score auf [0,1] (`_clamp_0_1()`)
   - **Zusätzlicher Fallback:** Wenn Score < 0.7 und keine Issues → `_fallback_issues()` (einfache Heuristiken: Wortanzahl, Kommata, Klammern)
4. **Final Score:** Direkt vom LLM (keine Aggregation/Weights)

---

## Komponenten

**LLM-basiert (primär):**
- Score: Direkt vom LLM-Output (v1: 0-1, v2: 1-5 → normalisiert)
- Issues: Vom LLM-Output (JSON-Array)
- Explanation: Vom LLM-Output

**Heuristik-Fallback (sekundär):**
- Issues nur, wenn LLM keine liefert und Score < 0.7
- Einfache Regeln: Wortanzahl ≥30, Kommata ≥4, Klammern vorhanden
- **Wichtig:** Heuristiken beeinflussen **nicht** den Score, nur Issues

---

## Code-Snippets

### LLM-Call (Primärquelle für Score)

```72:78:app/services/agents/readability/readability_verifier.py
    def evaluate(
        self,
        article_text: str,
        summary_text: str,
    ) -> Tuple[float, List[ReadabilityIssue], str]:
        prompt = self._build_prompt(article_text, summary_text)
        raw = self.llm.complete(prompt)
        data = self._parse_output(raw)
```

### Score-Extraktion (v1 vs v2)

```82:90:app/services/agents/readability/readability_verifier.py
        # v2: score kommt als integer 1-5, normalisiere auf [0,1]
        if self.prompt_version == "v2":
            score_raw = data.get("score_raw_1_to_5")
            if score_raw is not None:
                # Normalisiere: (score_raw - 1) / 4
                score = self._clamp01((float(score_raw) - 1.0) / 4.0)
            else:
                score = self._clamp01(data.get("score", 0.0))
        else:
            score = self._clamp01(data.get("score", 0.0))
```

### Heuristik-Fallback (nur für Issues, nicht Score)

```130:132:app/services/agents/readability/readability_verifier.py
        # Heuristik-Fallback: wenn score niedrig aber keine Issues
        if score < self.ISSUE_REQUIRED_BELOW and not issues:
            issues = self._heuristic_fallback_issues(summary_text)
```

### Final Score Clamp

```45:47:app/services/agents/readability/readability_agent.py
        score, issues, explanation = self.evaluator.evaluate(article_text, summary_text)

        score = self._clamp_0_1(score)
```

---

## ENV/Config-Flags

- **`READABILITY_PROMPT_VERSION`** (ENV): `"v1"` oder `"v2"` (default: `"v1"`)
- **`ISSUE_FALLBACK_THRESHOLD = 0.7`** (ReadabilityAgent): Schwelle für Fallback-Issues
- **`ISSUE_REQUIRED_BELOW = 0.7`** (LLMReadabilityEvaluator): Schwelle für Heuristik-Fallback

---

## Mitigation (Output-Collapse)

**Problem:** v2 zeigt Output-Collapse (fast alle Predictions bei ~0.25)

**Ursache:** Prompt v2 (1-5 integer score) führt zu eingeschränkter LLM-Ausgabe

**Mitigation (implementiert in eval_sumeval_readability.py):**
- Collapse-Detector: Wenn >80% der predictions im selben Bucket → Warnung in `summary.md` + `summary.json`
- Robustes JSON-Parsing mit Retry-Logik
- Distribution wird in `summary.md` dokumentiert

