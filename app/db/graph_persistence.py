from neo4j import Transaction
from app.db.neo4j_client import get_neo4j_session
from app.models.schemas import AgentResult
import logging

logger = logging.getLogger(__name__)


def write_verification_graph(
    article_id: int | str,
    summary_id: int | str,
    run_id: int | str,
    overall_score: float,
    factuality: AgentResult,
    coherence: AgentResult,
    readability: AgentResult,
) -> None:
    """Schreibt einen Verifikations-Run in Neo4j als Graphstruktur."""
    logger.info(
        "write_verification_graph START (article_id=%s, summary_id=%s, run_id=%s)",
        article_id,
        summary_id,
        run_id,
    )
    with get_neo4j_session() as session:
        session.execute_write(
            _write_graph_tx,
            article_id,
            summary_id,
            run_id,
            overall_score,
            factuality,
            coherence,
            readability,
        )
    logger.info("write_verification_graph DONE (run_id=%s)", run_id)


def _write_graph_tx(
    tx: Transaction,
    article_id: int | str,
    summary_id: int | str,
    run_id: int | str,
    overall_score: float,
    factuality: AgentResult,
    coherence: AgentResult,
    readability: AgentResult,
) -> None:
    # Article & Summary
    tx.run(
        """
        MERGE (a:Article {id: $article_id})
        MERGE (s:Summary {id: $summary_id})
        MERGE (a)-[:HAS_SUMMARY]->(s)
        """,
        article_id=article_id,
        summary_id=summary_id,
    )

    def metric_with_errors(dimension: str, agent: AgentResult) -> None:
        tx.run(
            """
            MATCH (s:Summary {id: $summary_id})
            MERGE (m:Metric {run_id: $run_id, dimension: $dimension})
            SET m.score = $score
            MERGE (s)-[:HAS_METRIC]->(m)
            """,
            summary_id=summary_id,
            run_id=run_id,
            dimension=dimension,
            score=agent.score,
        )

        for idx, err in enumerate(agent.errors):
            tx.run(
                """
                MATCH (s:Summary {id: $summary_id})
                MERGE (e:Error {
                    run_id: $run_id,
                    dimension: $dimension,
                    index: $idx
                })
                SET e.message = $message,
                    e.severity = $severity,
                    e.start_char = $start_char,
                    e.end_char = $end_char
                MERGE (s)-[:HAS_ERROR]->(e)
                """,
                summary_id=summary_id,
                run_id=run_id,
                dimension=dimension,
                idx=idx,
                message=err.message,
                severity=err.severity,
                start_char=err.start_char,
                end_char=err.end_char,
            )

    metric_with_errors("factuality", factuality)
    metric_with_errors("coherence", coherence)
    metric_with_errors("readability", readability)

    tx.run(
        """
        MATCH (s:Summary {id: $summary_id})
        MERGE (m:Metric {run_id: $run_id, dimension: 'overall'})
        SET m.score = $score
        MERGE (s)-[:HAS_METRIC]->(m)
        """,
        summary_id=summary_id,
        run_id=run_id,
        score=overall_score,
    )
