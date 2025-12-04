from fastapi import FastAPI, Depends
from app.api.routes import router as api_router
from app.core.config import settings
from app.db.session import get_db
from sqlalchemy import text
from app.db.neo4j_client import close_driver
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup-Phase
    # Falls du später etwas wie "initialize_embeddings()" o. ä. brauchst,
    # kommt es hier rein.
    yield
    # Shutdown-Phase
    close_driver()   # Neo4j sauber schließen

# Haupt-FastAPI-App
app = FastAPI(title="VerificationAPI", lifespan=lifespan,)

# bindet alle API-Routen ein
app.include_router(api_router)

# einfache Root-Route
@app.get("/")
async def root():
    return {"message": "Verification API running"}

# DB-Check auf App-Ebene (duplicated, aber ok für M1)
@app.get("/db-check")
def db_check(db=Depends(get_db)):
    result = db.execute(text("SELECT 1")).scalar()
    return {"db_ok": bool(result)}

@app.on_event("shutdown")
def on_shutdown():
    """
    Wird beim Herunterfahren der App ausgeführt.
    Schließt den Neo4j-Driver sauber.
    """
    close_driver()