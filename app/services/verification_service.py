import os

from sqlalchemy.orm import Session

from app.models.pydantic import VerifyRequest, PipelineResult
from app.pipeline.verification_pipeline import VerificationPipeline
from app.db.postgres.persistence import (
    store_article_and_summary,
    store_verification_run,
)

# Wenn TEST_MODE=1 in der Umgebung gesetzt ist,
# werden keine echten DB-Zugriffe ausgeführt.
TEST_MODE = os.getenv("TEST_MODE") == "1"


class VerificationService:
    def __init__(self) -> None:
        self.pipeline = VerificationPipeline()

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

        # 3. Run + Ergebnisse speichern (nur, wenn nicht im Test-Modus)
        if not TEST_MODE:
            run_id = store_verification_run(
                db=db,
                article_id=article_id,
                summary_id=summary_id,
                overall_score=result.overall_score,
                factuality=result.factuality,
                coherence=result.coherence,
                readability=result.readability,
            )
        else:
            # Im Testmodus keine DB, also Dummy-Run-ID
            run_id = -1

        return run_id, result
