from app.models.schemas import VerifyRequest, PipelineResult
from app.services.pipeline import VerificationPipeline
from app.db.persistence import (
    store_article_and_summary,
    store_verification_run,
)
from sqlalchemy.orm import Session


class VerificationService:
    def __init__(self) -> None:
        self.pipeline = VerificationPipeline()

    def verify(self, req: VerifyRequest, db: Session) -> tuple[int, PipelineResult]:
        # 1. Artikel + Summary speichern
        article_id, summary_id = store_article_and_summary(
            dataset=req.dataset,
            article_text=req.article_text,
            summary_text=req.summary_text,
            llm_model=req.llm_model,
        )

        # 2. Pipeline ausführen
        result = self.pipeline.run(
            article=req.article_text,
            summary=req.summary_text,
            meta=req.meta or {},
        )

        # 3. Run + Ergebnisse speichern (DB weitergeben!)
        run_id = store_verification_run(
            db=db,
            article_id=article_id,
            summary_id=summary_id,
            overall_score=result.overall_score,
            factuality=result.factuality,
            coherence=result.coherence,
            readability=result.readability,
        )

        return run_id, result
