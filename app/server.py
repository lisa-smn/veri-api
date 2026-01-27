from dotenv import load_dotenv
import os

load_dotenv()
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from sqlalchemy import text

from app.api.routes import router as api_router
from app.core.config import settings
from app.db.neo4j.neo4j_client import close_driver
from app.db.postgres.session import get_db


def validate_startup_config():
    """Validiert kritische Umgebungsvariablen beim Startup (fail-fast)."""
    errors = []
    
    # Prüfe OPENAI_API_KEY (wird für LLM-Calls benötigt)
    if not os.getenv("OPENAI_API_KEY"):
        errors.append(
            "OPENAI_API_KEY is not set. "
            "Set it in .env file or as environment variable. "
            "Required for LLM-based verification."
        )
    
    # Prüfe DATABASE_URL (wird für Postgres benötigt)
    # settings.database_url hat einen Default für Docker, aber sollte nicht leer sein
    database_url = os.getenv("DATABASE_URL") or settings.database_url
    if not database_url or database_url.strip() == "":
        errors.append(
            "DATABASE_URL is not set or empty. "
            "Set it in .env file or as environment variable. "
            "Required for database persistence."
        )
    
    if errors:
        error_msg = "Startup validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Validierung beim Startup
    validate_startup_config()
    yield
    close_driver()


app = FastAPI(title="VerificationAPI", lifespan=lifespan)
app.include_router(api_router)


@app.get("/")
async def root():
    return {"message": "Verification API running"}


@app.get("/db-check")
def db_check(db=Depends(get_db)):
    return {"db_ok": bool(db.execute(text("SELECT 1")).scalar())}
