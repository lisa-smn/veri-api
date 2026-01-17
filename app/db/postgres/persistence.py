import json
import logging

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.neo4j.graph_persistence import write_verification_graph
from app.db.postgres.session import SessionLocal
from app.models.pydantic import AgentResult
from app.services.explainability.explainability_models import ExplainabilityResult

logger = logging.getLogger(__name__)


def _pydantic_dump(obj) -> dict:
    """
    Pydantic v2/v1 kompatibler Dump.
    - v2: model_dump()
    - v1: dict()
    """
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        # Pydantic v2
        return obj.model_dump(mode="json")
    if hasattr(obj, "dict"):
        # Pydantic v1
        return obj.dict()
    # letzter Fallback: best effort
    return {"value": str(obj)}


def store_article_and_summary(
    dataset: str | None,
    article_text: str,
    summary_text: str,
    llm_model: str | None,
) -> tuple[int, int]:
    """
    Legt (falls nötig) einen Datensatz an, speichert Artikel + Summary
    und gibt (article_id, summary_id) zurück.
    """
    db: Session = SessionLocal()
    try:
        dataset_id = None

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
    explainability: ExplainabilityResult | None = None,
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
            spans_payload = []
            for s in agent.issue_spans:
                if hasattr(s, "model_dump") or hasattr(s, "dict"):
                    spans_payload.append(_pydantic_dump(s))
                else:
                    logger.warning("Non-IssueSpan in %s.issue_spans: %r", dimension, s)
                    spans_payload.append(
                        {
                            "start_char": None,
                            "end_char": None,
                            "message": str(s),
                            "severity": "low",
                            "issue_type": None,
                        }
                    )

            vr_result = db.execute(
                text(
                    """
                    INSERT INTO verification_results (run_id, dimension, score, explanation, issue_spans, details)
                    VALUES (:run_id, :dimension, :score, :explanation, CAST(:issue_spans AS jsonb), CAST(:details AS jsonb))
                    RETURNING id
                    """
                ),
                {
                    "run_id": run_id,
                    "dimension": dimension,
                    "score": agent.score,
                    "explanation": agent.explanation,
                    "issue_spans": json.dumps(spans_payload),
                    "details": json.dumps(agent.details) if agent.details is not None else None,
                },
            )
            vr_id = vr_result.scalar_one()

            # explanations-Tabelle: optionaler Audit-Trail
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
                        VALUES (:run_id, :vr_id, :agent_name, :explanation, CAST(:raw_response AS jsonb))
                        """
                    ),
                    {
                        "run_id": run_id,
                        "vr_id": vr_id,
                        "agent_name": agent_name,
                        "explanation": agent.explanation,
                        "raw_response": json.dumps(
                            {
                                "score": agent.score,
                                "issue_spans": spans_payload,
                                "details": agent.details,
                            }
                        ),
                    },
                )

        logger.debug("factuality issue_spans: %s", factuality.issue_spans)
        logger.debug("coherence issue_spans: %s", coherence.issue_spans)
        logger.debug("readability issue_spans: %s", readability.issue_spans)

        insert_dimension("factuality", factuality, "FactualityAgent")
        insert_dimension("coherence", coherence, "CoherenceAgent")
        insert_dimension("readability", readability, "ReadabilityAgent")

        # overall separat
        db.execute(
            text(
                """
                INSERT INTO verification_results (run_id, dimension, score, explanation, issue_spans, details)
                VALUES (:run_id, 'overall', :score, NULL, '[]'::jsonb, NULL)
                """
            ),
            {"run_id": run_id, "score": overall_score},
        )

        # Explainability Report speichern (falls vorhanden)
        if explainability is not None:
            try:
                report_payload = _pydantic_dump(explainability)

                db.execute(
                    text(
                        """
                        INSERT INTO explainability_reports (run_id, version, report_json)
                        VALUES (:run_id, :version, CAST(:report_json AS jsonb))
                        """
                    ),
                    {
                        "run_id": run_id,
                        "version": getattr(explainability, "version", "m9_v1"),
                        "report_json": json.dumps(report_payload),
                    },
                )
            except Exception as e:
                # Tabelle existiert möglicherweise nicht - optional, nicht kritisch
                logger.warning(
                    f"Explainability-Report konnte nicht gespeichert werden (Tabelle fehlt?): {e}"
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

    # Neo4j: best effort (sollte durch NEO4J_ENABLED ohnehin in Tests aus sein)
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
