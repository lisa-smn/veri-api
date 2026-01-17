"""
Orchestrator

- nimmt API-Request_Daten entgegen
- speichert Artikel + Summary in der DB
- führt die Verifikations-Pipeline aus
- speichert die Ergebnisse in der DB
- liefert die Run-ID und das Pipeline-Ergebnis zurück

"""

from collections import Counter
import os

from sqlalchemy.orm import Session

from app.db.postgres.persistence import (
    store_article_and_summary,
    store_verification_run,
)
from app.llm.openai_client import OpenAIClient
from app.models.pydantic import PipelineResult, VerifyRequest
from app.pipeline.verification_pipeline import VerificationPipeline
from app.services.judges.llm_judge import LLMJudge

# Wenn TEST_MODE=1 in der Umgebung gesetzt ist,
# werden keine echten DB-Zugriffe ausgeführt.
TEST_MODE = os.getenv("TEST_MODE") == "1"


class VerificationService:
    def __init__(self) -> None:
        self.pipeline = VerificationPipeline()
        # LLM-as-a-Judge (nur initialisieren wenn ENABLE_LLM_JUDGE aktiv)
        self.judge_enabled = os.getenv("ENABLE_LLM_JUDGE", "false").lower() == "true"
        if self.judge_enabled:
            llm_client = OpenAIClient()
            self.llm_judge = LLMJudge(
                llm_client=llm_client,
                default_model="gpt-4o-mini",
                default_prompt_version="v2_binary",  # Für Factuality binary
                default_n=3,
                default_temperature=0.0,
                default_aggregation="majority",
            )
        else:
            self.llm_judge = None

    def verify(self, req: VerifyRequest, db: Session) -> tuple[int, PipelineResult]:
        # Meta-Daten vorbereiten (falls du später Dataset/Model etc. da reinstecken willst)
        meta = getattr(req, "meta", None) or {}

        # 1. Artikel + Summary speichern (nur, wenn nicht im Test-Modus)
        if not TEST_MODE:
            article_id, summary_id = store_article_and_summary(
                dataset=req.dataset,
                article_text=req.article_text,
                summary_text=req.summary_text,
                llm_model=req.llm_model,
            )
        else:
            # Fake-IDs im Testmode, damit der Rückgabetyp gleich bleibt
            article_id, summary_id = -1, -1

        # 2. Pipeline ausführen (immer)
        result = self.pipeline.run(
            article=req.article_text,
            summary=req.summary_text,
            meta=meta,
        )

        # 3. LLM-as-a-Judge ausführen (optional, wenn aktiviert)
        judge_data = None
        judge_error = None
        judge_available = False

        if req.run_llm_judge and self.judge_enabled and self.llm_judge:
            try:
                # Factuality Judge (binary verdict + confidence)
                judge_mode = req.judge_mode or "primary"
                judge_n = req.judge_n or 3
                judge_temperature = (
                    req.judge_temperature if req.judge_temperature is not None else 0.0
                )
                judge_aggregation = req.judge_aggregation or "majority"

                judge_result = self.llm_judge.judge(
                    dimension="factuality",
                    article_text=req.article_text,
                    summary_text=req.summary_text,
                    model=None,  # Nutze default
                    prompt_version="v2_binary",
                    n=judge_n,
                    temperature=judge_temperature,
                    aggregation=judge_aggregation,
                )

                # Extrahiere binary verdict und confidence
                # Für Factuality: judge_result enthält outputs mit error_present und confidence in raw_json
                error_present = None
                confidence = None

                if judge_result.outputs:
                    # Aggregation: majority vote für error_present
                    # error_present ist in raw_json gespeichert
                    error_presents = []
                    for out in judge_result.outputs:
                        if out.raw_json and out.raw_json.get("error_present") is not None:
                            error_presents.append(out.raw_json.get("error_present"))
                        # Fallback: aus score_norm ableiten (0.0 = error, 1.0 = no error)
                        elif out.score_norm is not None:
                            error_presents.append(out.score_norm < 0.5)

                    if error_presents:
                        error_counts = Counter(error_presents)
                        error_present = error_counts.most_common(1)[0][0] if error_counts else None

                    # Mean confidence (aus raw_json oder direkt aus output.confidence)
                    confidences = []
                    for out in judge_result.outputs:
                        if out.confidence is not None:
                            confidences.append(out.confidence)
                        elif out.raw_json and isinstance(
                            out.raw_json.get("confidence"), (int, float)
                        ):
                            confidences.append(out.raw_json.get("confidence"))

                    if confidences:
                        confidence = sum(confidences) / len(confidences)

                judge_data = {
                    "factuality": {
                        "error_present": error_present,
                        "confidence": confidence,
                        "model": judge_result.outputs[0].model if judge_result.outputs else None,
                        "prompt_version": "v2_binary",
                        "judge_n": judge_n,
                        "aggregation": judge_aggregation,
                    }
                }
                judge_available = True
            except Exception as e:
                # Judge-Fehler soll Verify nicht killen
                judge_error = str(e)
                judge_available = False
        elif req.run_llm_judge and not self.judge_enabled:
            judge_error = "ENABLE_LLM_JUDGE not set in environment"
            judge_available = False

        # 4. Run + Ergebnisse speichern (nur, wenn nicht im Test-Modus)
        if not TEST_MODE:
            run_id = store_verification_run(
                db=db,
                article_id=article_id,
                summary_id=summary_id,
                overall_score=result.overall_score,
                factuality=result.factuality,
                coherence=result.coherence,
                readability=result.readability,
                explainability=result.explainability,
            )
        else:
            # Im Testmodus keine DB, also Dummy-Run-ID
            run_id = -1

        # 5. Judge-Daten an Result anhängen (für Response)
        result.judge = judge_data
        result.judge_error = judge_error
        result.judge_available = judge_available

        return run_id, result
