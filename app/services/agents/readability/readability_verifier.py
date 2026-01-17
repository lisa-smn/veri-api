"""
Dieses Modul implementiert den Readability-Verifier, der die eigentliche
Lesbarkeitsbewertung einer Summary durchführt.

Der Verifier kapselt die gesamte Bewertungslogik und stellt eine einheitliche
Schnittstelle (`ReadabilityEvaluator`) bereit, über die unterschiedliche
Implementierungen (z.B. LLM-basiert oder heuristisch) angebunden werden können.
Die Standardimplementierung (`LLMReadabilityEvaluator`) nutzt ein Sprachmodell,
um einen normalisierten Readability-Score, eine globale Erklärung sowie eine
strukturierte Liste von Readability-Issues zu erzeugen.

Der Verifier bewertet ausschließlich Aspekte der Lesbarkeit wie Satzlänge,
syntaktische Komplexität, Interpunktionsdichte und Lesefluss. Faktische
Korrektheit, logische Kohärenz sowie Stil oder Tonalität werden explizit
nicht berücksichtigt.

Die Ausgabe des Verifiers besteht aus:
- einem Score im Bereich [0,1]
- einer Liste strukturierter Readability-Issues (ohne Text-Offsets)
- einer kurzen, globalen Erklärung

Die Abbildung der erkannten Issues auf konkrete Textstellen (IssueSpans) sowie
die Einbettung in das standardisierte AgentResult erfolgen bewusst nicht im
Verifier, sondern im ReadabilityAgent. Dadurch bleibt der Verifier unabhängig
von Persistenz, Pipeline-Details und UI-relevanter Darstellung.

"""

from __future__ import annotations

import json
from typing import Any, Protocol

from app.llm.llm_client import LLMClient
from app.services.agents.readability.readability_models import ReadabilityIssue


class ReadabilityEvaluator(Protocol):
    def evaluate(
        self,
        article_text: str,
        summary_text: str,
    ) -> tuple[float, list[ReadabilityIssue], str]:
        """
        Returns:
          - score: float in [0, 1]
          - issues: List[ReadabilityIssue]
          - explanation: str
        """
        ...


class LLMReadabilityEvaluator:
    """
    LLM-basierter Evaluator für Readability.

    Wichtig:
    - Scope: nur Lesbarkeit (Satzlänge, Verschachtelung, Interpunktion, Parsbarkeit)
    - Nicht Scope: Fakten, Kohärenz, Stil/Tonalität, inhaltliche Qualität
    - Liefert Issues mit summary_span als wörtlicher Substring für Mapping
    - Falls das LLM keine Issues liefert, aber score niedrig ist:
      Heuristik-Fallback erzeugt deterministische Issues.
    """

    # Wenn Score darunter liegt, müssen wir i.d.R. Issues haben.
    ISSUE_REQUIRED_BELOW = 0.7

    def __init__(self, llm_client: LLMClient, prompt_version: str = "v1"):
        self.llm = llm_client
        self.prompt_version = prompt_version

    def evaluate(
        self,
        article_text: str,
        summary_text: str,
    ) -> tuple[float, list[ReadabilityIssue], str]:
        prompt = self._build_prompt(article_text, summary_text)
        raw = self.llm.complete(prompt)
        data = self._parse_output(raw)

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
        explanation = (data.get("explanation") or "").strip()

        issues: list[ReadabilityIssue] = []
        for item in data.get("issues", []):
            if not isinstance(item, dict):
                continue

            # harte Validierung (damit nichts Unerwartetes durchrutscht)
            t = item.get("type", "HARD_TO_PARSE")
            if t not in (
                "LONG_SENTENCE",
                "COMPLEX_NESTING",
                "PUNCTUATION_OVERLOAD",
                "HARD_TO_PARSE",
            ):
                t = "HARD_TO_PARSE"

            sev = item.get("severity", "medium")
            if sev not in ("low", "medium", "high"):
                sev = "medium"

            span = (item.get("summary_span") or "").strip()
            comment = (item.get("comment") or "").strip()

            metric = item.get("metric")
            if metric is not None and not isinstance(metric, str):
                metric = str(metric)

            metric_value = self._to_float_or_none(item.get("metric_value"))

            issues.append(
                ReadabilityIssue(
                    type=t,
                    severity=sev,
                    summary_span=span,
                    comment=comment,
                    metric=metric,
                    metric_value=metric_value,
                )
            )

        # Postprocess: summary_span muss substring sein, sonst fallback snippet
        issues = self._ensure_spans_are_substrings(summary_text, issues)

        # Heuristik-Fallback: wenn score niedrig aber keine Issues
        if score < self.ISSUE_REQUIRED_BELOW and not issues:
            issues = self._heuristic_fallback_issues(summary_text)

        # Fallback-Explanation, falls LLM leer liefert
        if not explanation:
            if issues:
                explanation = (
                    "Die Summary ist schwer lesbar; es wurden konkrete Lesbarkeitsprobleme erkannt."
                )
            else:
                explanation = "Die Summary wirkt insgesamt gut lesbar."

        return score, issues, explanation

    # -------------------------
    # Prompt
    # -------------------------

    def _build_prompt(self, article: str, summary: str) -> str:
        if self.prompt_version == "v2":
            return self._build_prompt_v2(article, summary)
        return self._build_prompt_v1(article, summary)

    def _build_prompt_v1(self, article: str, summary: str) -> str:
        # Wichtig: Readability ≠ Stil/Ton, ≠ Fakten, ≠ Kohärenz
        return f"""
Du bewertest NUR die LESBARKEIT (Readability) einer Summary.

Bewerte KEINE:
- faktische Korrektheit
- logische Kohärenz
- Stil, Tonalität, "Schönheit" der Sprache
- inhaltliche Qualität

Kriterien für Readability:
- Lesefluss und Verständlichkeit
- überlange Sätze
- unnötige Verschachtelung / zu viele Nebensätze
- Interpunktions-Überladung (z.B. extrem viele Kommata, Klammern)
- schwer zu parsende Satzkonstruktionen

Gib NUR JSON zurück, ohne zusätzliche Erklärungen außerhalb des JSON.

Schema:
{{
  "score": 0.0,  # float in [0,1] (1 = sehr gut lesbar, 0 = sehr schlecht lesbar)
  "explanation": "kurze globale Begründung",
  "issues": [
    {{
      "type": "LONG_SENTENCE" | "COMPLEX_NESTING" | "PUNCTUATION_OVERLOAD" | "HARD_TO_PARSE",
      "severity": "low" | "medium" | "high",
      "summary_span": "wörtlicher Auszug aus der Summary (kurz, exakt kopiert)",
      "comment": "kurze Begründung, was daran die Lesbarkeit senkt",
      "metric": "optional: z.B. word_count | comma_count | nesting_depth",
      "metric_value": 0.0  # optional
    }}
  ]
}}

WICHTIG:
- summary_span MUSS direkt aus der Summary kopiert sein (damit wir die Stelle mappen können).
- Maximal 8 issues (wähle die wichtigsten).
- Wenn score < 0.7, gib MINDESTENS 1 issue zurück (issues darf dann nicht leer sein).
- Wenn du keine perfekte Stelle findest: nutze als summary_span die ersten 80–120 Zeichen der Summary (exakt kopiert).

ARTIKEL (nur Kontext, nicht bewerten):
{article}

SUMMARY (zu bewerten):
{summary}
""".strip()

    def _build_prompt_v2(self, article: str, summary: str) -> str:
        """Prompt v2: Klare Rubrik, 1-5 integer score, volle Skala nutzen."""
        return f"""
Du bewertest NUR die LESBARKEIT (Readability) einer Summary.

Bewerte KEINE:
- faktische Korrektheit
- logische Kohärenz
- Stil, Tonalität, "Schönheit" der Sprache
- inhaltliche Qualität

**Rubrik für Readability (Skala 1-5, 5 = exzellent, 1 = sehr schlecht):**

1. **Grammar & Syntax (Grammatik & Syntax):**
   - Korrekte Grammatik, keine Syntaxfehler
   - Klare Satzstruktur, keine unvollständigen Sätze

2. **Clarity (Klarheit):**
   - Eindeutige Wortwahl, keine mehrdeutigen Begriffe
   - Klare Referenzen (Pronomen, Bezüge)

3. **Flow (Lesefluss):**
   - Flüssige Übergänge zwischen Sätzen
   - Angemessene Satzlänge (nicht zu lang, nicht zu kurz)
   - Keine abrupten Sprünge

4. **Conciseness (Prägnanz):**
   - Keine unnötige Verschachtelung
   - Keine Interpunktions-Überladung (z.B. extrem viele Kommata, Klammern)
   - Keine schwer zu parsende Satzkonstruktionen

**WICHTIG: Nutze die volle Skala 1-5!**
- 5 = exzellent lesbar (keine Probleme)
- 4 = gut lesbar (kleine Probleme)
- 3 = akzeptabel (moderate Probleme)
- 2 = schwer lesbar (deutliche Probleme)
- 1 = sehr schwer lesbar (schwere Probleme)

Gib NUR JSON zurück, ohne zusätzliche Erklärungen außerhalb des JSON.

Schema:
{{
  "score_raw_1_to_5": 3,  # INTEGER 1-5 (NICHT float, NICHT 0-1)
  "rationale": "kurze Begründung (max 2 Sätze), warum dieser Score",
  "explanation": "kurze globale Begründung",
  "issues": [
    {{
      "type": "LONG_SENTENCE" | "COMPLEX_NESTING" | "PUNCTUATION_OVERLOAD" | "HARD_TO_PARSE",
      "severity": "low" | "medium" | "high",
      "summary_span": "wörtlicher Auszug aus der Summary (kurz, exakt kopiert)",
      "comment": "kurze Begründung, was daran die Lesbarkeit senkt",
      "metric": "optional: z.B. word_count | comma_count | nesting_depth",
      "metric_value": 0.0  # optional
    }}
  ]
}}

WICHTIG:
- score_raw_1_to_5 MUSS ein INTEGER zwischen 1 und 5 sein (kein float, kein 0-1 Wert).
- summary_span MUSS direkt aus der Summary kopiert sein (damit wir die Stelle mappen können).
- Maximal 8 issues (wähle die wichtigsten).
- Wenn score_raw_1_to_5 <= 2, gib MINDESTENS 1 issue zurück (issues darf dann nicht leer sein).
- Wenn du keine perfekte Stelle findest: nutze als summary_span die ersten 80–120 Zeichen der Summary (exakt kopiert).

ARTIKEL (nur Kontext, nicht bewerten):
{article}

SUMMARY (zu bewerten):
{summary}
""".strip()

    # -------------------------
    # Parsing (robust)
    # -------------------------

    def _parse_output(self, raw: str) -> dict[str, Any]:
        """
        Robust wie bei deinem Claim-Verifier:
        - zieht erstes JSON-Objekt aus der Antwort
        - validiert Felder
        - liefert fallback bei Parsefehler
        """
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            json_str = raw[start:end]
            data = json.loads(json_str)
        except Exception:
            return {
                "score": 0.0,
                "explanation": f"Antwort nicht parsbar: {raw[:200]}",
                "issues": [],
            }

        score = self._to_float_or_default(data.get("score", 0.0), 0.0)
        explanation = (data.get("explanation") or "").strip()

        issues = data.get("issues") or []
        if not isinstance(issues, list):
            issues = []

        norm_issues: list[dict[str, Any]] = []
        for it in issues[:8]:
            if not isinstance(it, dict):
                continue

            t = it.get("type", "HARD_TO_PARSE")
            if t not in (
                "LONG_SENTENCE",
                "COMPLEX_NESTING",
                "PUNCTUATION_OVERLOAD",
                "HARD_TO_PARSE",
            ):
                t = "HARD_TO_PARSE"

            sev = it.get("severity", "medium")
            if sev not in ("low", "medium", "high"):
                sev = "medium"

            span = (it.get("summary_span") or "").strip()
            comment = (it.get("comment") or "").strip()

            metric = it.get("metric")
            if metric is not None and not isinstance(metric, str):
                metric = str(metric)

            metric_value = self._to_float_or_none(it.get("metric_value"))

            norm_issues.append(
                {
                    "type": t,
                    "severity": sev,
                    "summary_span": span,
                    "comment": comment,
                    "metric": metric,
                    "metric_value": metric_value,
                }
            )

        result = {
            "score": self._clamp01(score),
            "explanation": explanation,
            "issues": norm_issues,
        }
        # v2: speichere auch den raw score 1-5
        if self.prompt_version == "v2":
            score_raw = data.get("score_raw_1_to_5")
            if score_raw is not None:
                result["score_raw_1_to_5"] = int(score_raw)
        return result

    # -------------------------
    # Postprocessing / Fallbacks
    # -------------------------

    def _ensure_spans_are_substrings(
        self, summary_text: str, issues: list[ReadabilityIssue]
    ) -> list[ReadabilityIssue]:
        """
        Garantiert: summary_span ist entweder leer oder ein echter Substring.
        Wenn nicht: ersetzt durch ein sicheres Snippet (exakt aus Summary).
        """
        safe_snippet = summary_text.strip()[:120] if summary_text else ""
        out: list[ReadabilityIssue] = []

        for iss in issues:
            span = (iss.summary_span or "").strip()
            if span and span not in summary_text:
                iss.summary_span = safe_snippet
            elif not span and safe_snippet:
                # wenn LLM leer liefert, aber wir irgendwas mappen wollen
                iss.summary_span = safe_snippet
            out.append(iss)

        return out

    def _heuristic_fallback_issues(self, summary_text: str) -> list[ReadabilityIssue]:
        """
        Deterministische Fallback-Issues, wenn LLM keine Issues liefert.
        Ziel: issue_spans sind bei schlechtem Text nicht leer.
        """
        text = (summary_text or "").strip()
        if not text:
            return [
                ReadabilityIssue(
                    type="HARD_TO_PARSE",
                    severity="high",
                    summary_span="",
                    comment="Die Summary ist leer oder enthält keinen verwertbaren Text.",
                    metric=None,
                    metric_value=None,
                )
            ]

        words = text.split()
        word_count = len(words)
        comma_count = text.count(",")
        has_parentheses = "(" in text and ")" in text

        issues: list[ReadabilityIssue] = []

        # LONG_SENTENCE
        if word_count >= 30:
            span = " ".join(words[: min(25, word_count)])
            issues.append(
                ReadabilityIssue(
                    type="LONG_SENTENCE",
                    severity="high" if word_count >= 45 else "medium",
                    summary_span=span,  # garantiert Substring
                    comment=f"Sehr lange Satzstruktur ({word_count} Wörter) erschwert das Lesen.",
                    metric="word_count",
                    metric_value=float(word_count),
                )
            )

        # PUNCTUATION_OVERLOAD
        if comma_count >= 4:
            span = text[: min(120, len(text))]
            issues.append(
                ReadabilityIssue(
                    type="PUNCTUATION_OVERLOAD",
                    severity="medium",
                    summary_span=span,
                    comment=f"Viele Kommata ({comma_count}) deuten auf starke Verschachtelung hin.",
                    metric="comma_count",
                    metric_value=float(comma_count),
                )
            )

        # COMPLEX_NESTING (Klammer-Einschub als proxy)
        if has_parentheses:
            start = text.find("(")
            end = text.find(")", start + 1)
            if start != -1 and end != -1 and end > start:
                span = text[start : min(end + 1, len(text))]
            else:
                span = text[: min(120, len(text))]
            issues.append(
                ReadabilityIssue(
                    type="COMPLEX_NESTING",
                    severity="low",
                    summary_span=span,
                    comment="Einschübe/Klammern erhöhen syntaktische Komplexität.",
                    metric="nesting_depth",
                    metric_value=1.0,
                )
            )

        # Wenn nichts triggert, trotzdem 1 Issue
        if not issues:
            issues.append(
                ReadabilityIssue(
                    type="HARD_TO_PARSE",
                    severity="medium",
                    summary_span=text[: min(120, len(text))],
                    comment="Die Summary wirkt schwer lesbar, konnte aber nicht weiter aufgeschlüsselt werden.",
                    metric=None,
                    metric_value=None,
                )
            )

        # max 8
        return issues[:8]

    # -------------------------
    # Utils
    # -------------------------

    @staticmethod
    def _clamp01(x: Any) -> float:
        try:
            x = float(x)
        except Exception:
            return 0.0
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x

    @staticmethod
    def _to_float_or_default(value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _to_float_or_none(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None
