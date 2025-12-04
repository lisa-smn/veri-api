from typing import Optional

import json
import logging

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.schemas import AgentResult
from app.db.graph_persistence import write_verification_graph

logger = logging.getLogger(__name__)


def store_article_and_summary(
    dataset: Optional[str],
    article_text: str,
    summary_text: str,
    llm_model: Optional[str],
) -> tuple[int, int]:
    """
    Legt (falls nötig) einen Datensatz an, speichert Artikel + Summary
    und gibt (article_id, summary_id) zurück.
    """
    db: Session = SessionLocal()
    try:
        dataset_id = None

        # optional: Dataset per Name anlegen/finden
        if dataset:
            result = db.execute(
                text("SELECT id FROM datasets WHERE name = :name LIMIT 1"),
                {"name": dataset},
            ).first()
            if result:
                dataset_id = result[0]
            else:
                result = db.execute(
                    text(
                        """
                        INSERT INTO datasets (name)
                        VALUES (:name)
                        RETURNING id
                        """
                    ),
                    {"name": dataset},
                )
                dataset_id = result.scalar_one()

        # Artikel speichern
        article_result = db.execute(
            text(
                """
                INSERT INTO articles (dataset_id, text)
                VALUES (:dataset_id, :text)
                RETURNING id
                """
            ),
            {"dataset_id": dataset_id, "text": article_text},
        )
        article_id = article_result.scalar_one()

        # Summary speichern (source: 'llm')
        summary_result = db.execute(
            text(
                """
                INSERT INTO summaries (article_id, source, text, llm_model)
                VALUES (:article_id, 'llm', :text, :llm_model)
                RETURNING id
                """
            ),
            {
                "article_id": article_id,
                "text": summary_text,
                "llm_model": llm_model,
            },
        )
        summary_id = summary_result.scalar_one()

        db.commit()
        return article_id, summary_id

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def store_verification_run(
    db: Session,
    article_id: int,
    summary_id: int,
    overall_score: float,
    factuality: AgentResult,
    coherence: AgentResult,
    readability: AgentResult,
) -> int:
    """
    Legt einen Run an und speichert die Agenten-Ergebnisse in verification_results
    + explanations und aktualisiert anschließend best-effort den Neo4j-Graph.
    """

    try:
        run_result = db.execute(
            text(
                """
                INSERT INTO runs (article_id, summary_id, run_type, status)
                VALUES (:article_id, :summary_id, 'verification', 'success')
                RETURNING id
                """
            ),
            {"article_id": article_id, "summary_id": summary_id},
        )
        run_id = run_result.scalar_one()

        def insert_dimension(dimension: str, agent: AgentResult, agent_name: str) -> None:
            vr_result = db.execute(
                text(
                    """
                    INSERT INTO verification_results (run_id, dimension, score, details)
                    VALUES (:run_id, :dimension, :score, :details)
                    RETURNING id
                    """
                ),
                {
                    "run_id": run_id,
                    "dimension": dimension,
                    "score": agent.score,
                    "details": json.dumps({
                        "errors": [e.model_dump() for e in agent.errors],
                        "explanation": agent.explanation,
                    }),
                },
            )
            vr_id = vr_result.scalar_one()

            if agent.explanation:
                db.execute(
                    text(
                        """
                        INSERT INTO explanations (
                            run_id,
                            verification_result_id,
                            agent_name,
                            explanation,
                            raw_response
                        )
                        VALUES (:run_id, :vr_id, :agent_name, :explanation, :raw_response)
                        """
                    ),
                    {
                        "run_id": run_id,
                        "vr_id": vr_id,
                        "agent_name": agent_name,
                        "explanation": agent.explanation,
                        "raw_response": json.dumps({
                            "score": agent.score,
                            "errors": [e.model_dump() for e in agent.errors],
                        }),
                    },
                )

        print("DEBUG factuality errors:", [e.model_dump() for e in factuality.errors])
        print("DEBUG coherence errors:", [e.model_dump() for e in coherence.errors])
        print("DEBUG readability errors:", [e.model_dump() for e in readability.errors])

        insert_dimension("factuality", factuality, "FactualityAgent")
        insert_dimension("coherence", coherence, "CoherenceAgent")
        insert_dimension("readability", readability, "ReadabilityAgent")

        db.execute(
            text(
                """
                INSERT INTO verification_results (run_id, dimension, score)
                VALUES (:run_id, 'overall', :score)
                """
            ),
            {"run_id": run_id, "score": overall_score},
        )

        db.commit()

    except Exception:
        db.rollback()
        logger.exception(
            "Fehler beim Speichern des Verifikations-Runs in der Datenbank (article_id=%s, summary_id=%s)",
            article_id,
            summary_id,
        )
        raise

    # Neo4j: best effort
    try:
        write_verification_graph(
            article_id=article_id,
            summary_id=summary_id,
            run_id=run_id,
            overall_score=overall_score,
            factuality=factuality,
            coherence=coherence,
            readability=readability,
        )
    except Exception:
        logger.exception(
            "Fehler beim Schreiben des Verifikations-Graphs in Neo4j (run_id=%s)",
            run_id,
        )

    return run_id
