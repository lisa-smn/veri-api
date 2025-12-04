from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.models.schemas import VerifyRequest, VerifyResponse
from app.services.verification_service import VerificationService
from app.db.session import get_db
from sqlalchemy import text

router = APIRouter()
verification_service = VerificationService()

# einfacher Health-Check
@router.get("/health")
async def health():
    return {"status": "ok"}


# nimmt eine Anfrage entgegen und startet die Verifikation
@router.post("/verify", response_model=VerifyResponse)
def verify(req: VerifyRequest, db: Session = Depends(get_db)):
    try:
        # DB-Session an den Service weitergeben
        run_id, result = verification_service.verify(req, db)
        return VerifyResponse(
            run_id=run_id,
            overall_score=result.overall_score,
            factuality=result.factuality,
            coherence=result.coherence,
            readability=result.readability,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# prüft, ob die DB erreichbar ist
@router.get("/db-check")
def db_check(db = Depends(get_db)):
    result = db.execute(text("SELECT 1")).scalar()
    return {"db_ok": bool(result)}
