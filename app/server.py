from dotenv import load_dotenv

load_dotenv()
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from sqlalchemy import text

from app.api.routes import router as api_router
from app.db.neo4j.neo4j_client import close_driver
from app.db.postgres.session import get_db


@asynccontextmanager
async def lifespan(app: FastAPI):
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
