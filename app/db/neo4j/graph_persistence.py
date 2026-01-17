import logging
import os

from neo4j import Transaction

from app.db.neo4j.neo4j_client import get_neo4j_session
from app.models.pydantic import AgentResult

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
    """Schreibt einen Verifikations-Run in Neo4j als Graphstruktur (best-effort)."""

    if os.getenv("NEO4J_ENABLED", "1") != "1":
        return

    logger.info(
        "write_verification_graph START (article_id=%s, summary_id=%s, run_id=%s)",
        article_id,
        summary_id,
        run_id,
    )
    try:
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
    except Exception:
        logger.exception("write_verification_graph FAILED (run_id=%s)", run_id)
        return

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
    # --- Core Nodes ---
    tx.run(
        """
        MERGE (a:Article {id: $article_id})
        MERGE (s:Summary {id: $summary_id})
        MERGE (a)-[:HAS_SUMMARY]->(s)
        """,
        article_id=article_id,
        summary_id=summary_id,
    )

    # Run-Knoten mit run_id Property (für Cross-Store Konsistenz)
    tx.run(
        """
        MERGE (r:Run {id: $run_id})
        SET r.run_id = $run_id
        WITH r
        MATCH (s:Summary {id: $summary_id})
        MERGE (r)-[:EVALUATES]->(s)
        """,
        run_id=run_id,
        summary_id=summary_id,
    )

    def metric_with_issue_spans(dimension: str, agent: AgentResult) -> None:
        # Metric-Node eindeutig machen (run + summary + dimension)
        tx.run(
            """
            MATCH (s:Summary {id: $summary_id})
            MERGE (m:Metric {run_id: $run_id, summary_id: $summary_id, dimension: $dimension})
            SET m.score = $score
            MERGE (s)-[:HAS_METRIC]->(m)
            """,
            summary_id=summary_id,
            run_id=run_id,
            dimension=dimension,
            score=float(agent.score) if agent and agent.score is not None else 0.0,
        )

        spans = getattr(agent, "issue_spans", None) or []
        for idx, span in enumerate(spans):
            # robust: falls irgendwo noch Legacy-Strings auftauchen
            message = getattr(span, "message", str(span))
            severity = getattr(span, "severity", None)
            start_char = getattr(span, "start_char", None)
            end_char = getattr(span, "end_char", None)

            tx.run(
                """
                MATCH (m:Metric {run_id: $run_id, summary_id: $summary_id, dimension: $dimension})

                // Dual-Label: IssueSpan UND Error → beide Query-Welten funktionieren
                MERGE (sp:IssueSpan:Error {
                    run_id: $run_id,
                    summary_id: $summary_id,
                    dimension: $dimension,
                    span_index: $idx
                })
                SET sp.message = $message,
                    sp.severity = $severity,
                    sp.start_char = $start_char,
                    sp.end_char = $end_char

                MERGE (m)-[:HAS_ISSUE_SPAN]->(sp)
                """,
                run_id=run_id,
                summary_id=summary_id,
                dimension=dimension,
                idx=idx,
                message=message,
                severity=severity,
                start_char=start_char,
                end_char=end_char,
            )

    metric_with_issue_spans("factuality", factuality)
    metric_with_issue_spans("coherence", coherence)
    metric_with_issue_spans("readability", readability)

    # overall Metric (ohne spans)
    tx.run(
        """
        MATCH (s:Summary {id: $summary_id})
        MERGE (m:Metric {run_id: $run_id, summary_id: $summary_id, dimension: 'overall'})
        SET m.score = $score
        MERGE (s)-[:HAS_METRIC]->(m)
        """,
        summary_id=summary_id,
        run_id=run_id,
        score=float(overall_score) if overall_score is not None else 0.0,
    )
