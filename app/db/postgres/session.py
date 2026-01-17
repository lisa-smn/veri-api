from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

DATABASE_URL = settings.database_url  # DB-URL aus den App-Settings

# erstellt die Engine für SQLAlchemy
engine = create_engine(DATABASE_URL, future=True, echo=False)

# Session-Factory für einzelne Requests
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# liefert eine DB-Session und räumt danach wieder auf
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
