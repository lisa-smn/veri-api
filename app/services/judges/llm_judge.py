"""
Generisches LLM-as-a-Judge Modul.

Unterstützt:
- Readability, Coherence, Factuality
- Strict JSON Output
- Committee (mehrere Judgements)
- Robustes Parsing mit Retry
- Degeneration Guardrails
"""

import statistics
from typing import Literal

from app.llm.llm_client import LLMClient
from app.models.pydantic import CommitteeStats, JudgeOutput, JudgeResult
from app.services.judges.parsing import (
    check_degeneration,
    normalize_rating,
    parse_judge_json,
)
from app.services.judges.prompts import (
    build_coherence_prompt,
    build_factuality_prompt,
    build_readability_prompt,
)


class LLMJudge:
    """
    Generischer LLM-as-a-Judge für Textqualitätsbewertung.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        default_model: str = "gpt-4o-mini",
        default_prompt_version: str = "v1",
        default_n: int = 1,
        default_temperature: float = 0.0,
        default_aggregation: Literal["mean", "median", "majority"] = "mean",
    ):
        self.llm = llm_client
        self.default_model = default_model
        self.default_prompt_version = default_prompt_version
        self.default_n = default_n
        self.default_temperature = default_temperature
        self.default_aggregation = default_aggregation

    def judge(
        self,
        dimension: Literal["readability", "coherence", "factuality"],
        article_text: str,
        summary_text: str,
        *,
        model: str | None = None,
        prompt_version: str | None = None,
        n: int | None = None,
        temperature: float | None = None,
        aggregation: Literal["mean", "median", "majority"] | None = None,
        cache_key: str | None = None,
        retries: int = 2,
    ) -> JudgeResult:
        """
        Führt LLM-as-a-Judge Bewertung durch.

        Args:
            dimension: Zu bewertende Dimension
            article_text: Artikel-Text (für Factuality erforderlich, für andere optional)
            summary_text: Zusammenfassung
            model: LLM-Modell (default: self.default_model)
            prompt_version: Prompt-Version (default: self.default_prompt_version)
            n: Anzahl Judgements für Committee (default: self.default_n)
            temperature: LLM-Temperatur (default: self.default_temperature)
            aggregation: Aggregationsmethode (default: self.default_aggregation)
            cache_key: Optionaler Cache-Key (noch nicht implementiert)
            retries: Anzahl Retries bei Parse-Fehlern (default: 2)

        Returns:
            JudgeResult mit outputs, committee stats, final_score_norm
        """
        model = model or self.default_model
        prompt_version = prompt_version or self.default_prompt_version
        n = n or self.default_n
        temperature = temperature if temperature is not None else self.default_temperature
        aggregation = aggregation or self.default_aggregation

        # Baue Prompt
        prompt = self._build_prompt(dimension, article_text, summary_text, prompt_version)

        # Führe n Judgements durch
        outputs: list[JudgeOutput] = []
        for i in range(n):
            output = self._single_judgement(
                dimension=dimension,
                prompt=prompt,
                model=model,
                prompt_version=prompt_version,
                retries=retries,
            )
            if output:
                outputs.append(output)

        if not outputs:
            # Fallback: Wenn alle Judgements fehlschlagen
            return JudgeResult(
                outputs=[],
                committee=None,
                final_score_norm=0.5,  # Neutraler Fallback
                aggregation=aggregation,
            )

        # Berechne Committee-Stats
        committee = None
        if len(outputs) > 1:
            scores = [out.score_norm for out in outputs]
            committee = CommitteeStats(
                n=len(scores),
                mean=statistics.mean(scores),
                std=statistics.stdev(scores) if len(scores) > 1 else None,
                min=min(scores),
                max=max(scores),
            )

        # Aggregiere final_score_norm
        final_score = self._aggregate_scores(outputs, aggregation)

        # Prüfe auf Degeneration (konvertiere zu Dict für check_degeneration)
        degeneration_flags = check_degeneration(
            [out.model_dump() for out in outputs],
            dimension,
        )
        if degeneration_flags:
            # Füge Flags zu allen Outputs hinzu
            for out in outputs:
                out.flags.extend(degeneration_flags)

        return JudgeResult(
            outputs=outputs,
            committee=committee,
            final_score_norm=final_score,
            aggregation=aggregation,
        )

    def _build_prompt(
        self,
        dimension: Literal["readability", "coherence", "factuality"],
        article_text: str,
        summary_text: str,
        prompt_version: str,
    ) -> str:
        """Baut Prompt für gegebene Dimension."""
        if dimension == "readability":
            return build_readability_prompt(summary_text, article_text, prompt_version)
        if dimension == "coherence":
            return build_coherence_prompt(summary_text, article_text, prompt_version)
        if dimension == "factuality":
            if not article_text:
                raise ValueError("Factuality-Bewertung benötigt article_text")
            return build_factuality_prompt(summary_text, article_text, prompt_version)
        raise ValueError(f"Unbekannte Dimension: {dimension}")

    def _single_judgement(
        self,
        dimension: Literal["readability", "coherence", "factuality"],
        prompt: str,
        model: str,
        prompt_version: str,
        retries: int,
    ) -> JudgeOutput | None:
        """
        Führt ein einzelnes Judgement durch (mit Retry-Logik).

        Returns:
            JudgeOutput oder None bei komplettem Fehler
        """
        last_error = None

        for attempt in range(retries + 1):
            try:
                # LLM-Call
                raw_text = self.llm.complete(prompt)

                # Parse JSON
                parsed_data, flags = parse_judge_json(
                    raw_text,
                    dimension,
                    expected_schema={
                        "rating": float,
                        "score": float,
                        "rationale": str,
                        "confidence": float,
                    },
                )

                if not parsed_data:
                    if attempt < retries:
                        flags.append("parse_retry")
                        continue
                    # Kompletter Fehler
                    return None

                # Extrahiere Werte (unterstütze rating, score, oder error_present)
                rating_raw = parsed_data.get("rating")
                score_raw = parsed_data.get("score")
                error_present = parsed_data.get("error_present")

                # Binary factuality: error_present -> score_norm (0.0 = error, 1.0 = no error)
                if error_present is not None and dimension == "factuality":
                    # error_present: True -> score_norm = 0.0 (Fehler), False -> score_norm = 1.0 (kein Fehler)
                    score_norm = 0.0 if error_present else 1.0
                    rating_raw = 0 if error_present else 1  # Für JudgeOutput.rating_raw
                elif rating_raw is None and score_raw is None:
                    if attempt < retries:
                        flags.append("parse_retry")
                        continue
                    return None
                elif score_raw is not None:
                    # v2_float: score ist bereits 0-1
                    score_norm = normalize_rating(float(score_raw), rating_scale="0-1")
                    rating_raw = score_raw  # Für JudgeOutput.rating_raw
                    if score_norm != float(score_raw):
                        flags.append("score_clamped")  # Wurde geclampet
                else:
                    # v1: rating ist 1-5
                    score_norm = normalize_rating(float(rating_raw), rating_scale="1-5")

                # Erstelle JudgeOutput
                return JudgeOutput(
                    dimension=dimension,
                    rating_raw=rating_raw,
                    score_norm=score_norm,
                    rationale=parsed_data.get("rationale", "").strip(),
                    confidence=parsed_data.get("confidence"),
                    flags=flags,
                    model=model,
                    prompt_version=prompt_version,
                    raw_json=parsed_data,
                    raw_text=raw_text if "parse_fallback" in flags else None,
                )

            except Exception as e:
                last_error = str(e)
                if attempt < retries:
                    continue
                # Kompletter Fehler nach allen Retries
                return None

        return None

    def _aggregate_scores(
        self,
        outputs: list[JudgeOutput],
        aggregation: Literal["mean", "median", "majority"],
    ) -> float:
        """
        Aggregiert Scores aus mehreren Judgements.

        Args:
            outputs: Liste von JudgeOutputs
            aggregation: Aggregationsmethode

        Returns:
            Aggregierter Score [0,1]
        """
        if not outputs:
            return 0.5  # Neutraler Fallback

        scores = [out.score_norm for out in outputs]

        if aggregation == "mean":
            return statistics.mean(scores)
        if aggregation == "median":
            return statistics.median(scores)
        if aggregation == "majority":
            # Bei integer ratings: Mehrheitswert
            # Bei float: nächstgelegener Wert (vereinfacht)
            ratings = [out.rating_raw for out in outputs]
            if all(isinstance(r, int) for r in ratings):
                # Mehrheitswert bei integer ratings
                from collections import Counter

                most_common = Counter(ratings).most_common(1)[0][0]
                # Finde entsprechenden Output und normalisiere
                for out in outputs:
                    if out.rating_raw == most_common:
                        return out.score_norm
            # Fallback: Median
            return statistics.median(scores)
        raise ValueError(f"Unbekannte Aggregation: {aggregation}")
