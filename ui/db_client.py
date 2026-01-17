"""Postgres DB Client für Runs-Anzeige."""

import os
from typing import Any

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

# DSN aus ENV oder Fallback auf app.core.config
POSTGRES_DSN = os.getenv("POSTGRES_DSN")
if not POSTGRES_DSN:
    # Fallback: Baue DSN aus einzelnen ENV-Vars
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "veri_api")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "")
    if password:
        POSTGRES_DSN = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    else:
        POSTGRES_DSN = f"postgresql://{user}@{host}:{port}/{db}"


def get_engine() -> Engine | None:
    """Erstellt SQLAlchemy Engine, falls DSN verfügbar."""
    if not POSTGRES_DSN:
        return None
    try:
        return create_engine(POSTGRES_DSN, pool_pre_ping=True)
    except Exception:
        return None


def is_available() -> bool:
    """Prüft ob Postgres erreichbar ist."""
    engine = get_engine()
    if not engine:
        return False
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def get_latest_runs(limit: int = 50) -> list[dict[str, Any]]:
    """Holt die neuesten Runs aus Postgres."""
    engine = get_engine()
    if not engine:
        return []

    try:
        with engine.connect() as conn:
            # Prüfe ob runs-Tabelle existiert
            inspector = inspect(engine)
            if "runs" not in inspector.get_table_names():
                return []

            # Hole Runs mit optionalen Joins zu verification_results
            query = text("""
                SELECT 
                    r.id as run_id,
                    r.started_at as created_at,
                    r.status,
                    r.config,
                    COUNT(vr.id) as num_results
                FROM runs r
                LEFT JOIN verification_results vr ON vr.run_id = r.id
                GROUP BY r.id, r.started_at, r.status, r.config
                ORDER BY r.started_at DESC
                LIMIT :limit
            """)
            result = conn.execute(query, {"limit": limit})
            rows = result.fetchall()

            runs = []
            for row in rows:
                runs.append(
                    {
                        "run_id": row[0],
                        "created_at": row[1].isoformat() if row[1] else None,
                        "status": row[2],
                        "config": row[3] if row[3] else {},
                        "num_results": row[4],
                    }
                )
            return runs
    except Exception:
        return []


def get_run_details(run_id: int) -> dict[str, Any] | None:
    """Holt Details zu einem spezifischen Run."""
    engine = get_engine()
    if not engine:
        return None

    try:
        with engine.connect() as conn:
            # Run-Info
            run_query = text("""
                SELECT id, article_id, summary_id, run_type, status, config, started_at
                FROM runs
                WHERE id = :run_id
            """)
            run_row = conn.execute(run_query, {"run_id": run_id}).first()
            if not run_row:
                return None

            # Verification Results
            results_query = text("""
                SELECT dimension, score, explanation, issue_spans, details
                FROM verification_results
                WHERE run_id = :run_id
            """)
            results_rows = conn.execute(results_query, {"run_id": run_id}).fetchall()

            # Explainability Report
            explainability_query = text("""
                SELECT version, report_json, created_at
                FROM explainability_reports
                WHERE run_id = :run_id
                ORDER BY created_at DESC
                LIMIT 1
            """)
            explainability_row = conn.execute(explainability_query, {"run_id": run_id}).first()

            return {
                "run_id": run_row[0],
                "article_id": run_row[1],
                "summary_id": run_row[2],
                "run_type": run_row[3],
                "status": run_row[4],
                "config": run_row[5] if run_row[5] else {},
                "created_at": run_row[6].isoformat() if run_row[6] else None,
                "results": [
                    {
                        "dimension": r[0],
                        "score": r[1],
                        "explanation": r[2],
                        "issue_spans": r[3] if r[3] else [],
                        "details": r[4] if r[4] else {},
                    }
                    for r in results_rows
                ],
                "explainability": {
                    "version": explainability_row[0],
                    "report_json": explainability_row[1],
                    "created_at": explainability_row[2].isoformat()
                    if explainability_row[2]
                    else None,
                }
                if explainability_row
                else None,
            }
    except Exception:
        return None
