#!/usr/bin/env python3
"""
Proof Script für Explainability-Persistenz.

Erzeugt eine fixe run_id, erzeugt Explainability (ohne LLM), speichert in Postgres + Neo4j,
und verifiziert die DB-Einträge.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from sqlalchemy import create_engine, inspect, text
    from sqlalchemy.engine import Engine
except ImportError:
    print("❌ FEHLER: sqlalchemy nicht installiert", file=sys.stderr)
    sys.exit(1)

try:
    from neo4j import GraphDatabase
except ImportError:
    print("❌ FEHLER: neo4j nicht installiert", file=sys.stderr)
    sys.exit(1)

from app.models.pydantic import AgentResult, IssueSpan
from app.services.explainability.explainability_models import ExplainabilityResult
from app.services.explainability.explainability_service import ExplainabilityService
from scripts.audit_persistence import (
    get_neo4j_driver,
    get_postgres_engine,
)

FIXTURES_DIR = ROOT / "tests" / "fixtures"


def load_fixture(name: str) -> dict:
    """Lädt ein Fixture-File."""
    path = FIXTURES_DIR / f"explainability_input_{name}.json"
    with path.open() as f:
        return json.load(f)


def create_mock_agent_results(fixture_data: dict) -> tuple[AgentResult, AgentResult, AgentResult]:
    """Erstellt AgentResult-Objekte aus Fixture-Daten (ohne LLM)."""
    factuality_data = fixture_data.get("factuality", {})
    coherence_data = fixture_data.get("coherence", {})
    readability_data = fixture_data.get("readability", {})

    def build_agent_result(data: dict) -> AgentResult:
        issue_spans = []
        for span_data in data.get("issue_spans", []):
            issue_spans.append(IssueSpan(**span_data))

        return AgentResult(
            name=data.get("name", "agent"),
            score=data.get("score", 0.5),
            explanation=data.get("explanation", ""),
            issue_spans=issue_spans,
            details=data.get("details"),
        )

    factuality = build_agent_result(factuality_data)
    coherence = build_agent_result(coherence_data)
    readability = build_agent_result(readability_data)

    return factuality, coherence, readability


def store_proof_run_postgres(
    engine: Engine,
    run_id: str,
    article_text: str,
    summary_text: str,
    explainability: ExplainabilityResult,
) -> tuple[bool, int | None, str | None]:
    """
    Speichert Proof-Run in Postgres.
    Returns: (success, pg_run_id, error_message)
    """
    """Speichert Proof-Run in Postgres."""
    try:
        with engine.connect() as conn:
            # 1. Dataset
            dataset_result = conn.execute(
                text("""
                    INSERT INTO datasets (name, description)
                    VALUES ('proof', 'Proof test dataset')
                    ON CONFLICT DO NOTHING
                    RETURNING id
                """)
            )
            dataset_id = dataset_result.scalar_one() if dataset_result.rowcount > 0 else None
            if not dataset_id:
                dataset_result = conn.execute(
                    text("SELECT id FROM datasets WHERE name = 'proof' LIMIT 1")
                )
                dataset_id = dataset_result.scalar_one()

            # 2. Article
            article_result = conn.execute(
                text("""
                    INSERT INTO articles (dataset_id, text)
                    VALUES (:dataset_id, :text)
                    RETURNING id
                """),
                {"dataset_id": dataset_id, "text": article_text},
            )
            article_id = article_result.scalar_one()

            # 3. Summary
            summary_result = conn.execute(
                text("""
                    INSERT INTO summaries (article_id, source, text)
                    VALUES (:article_id, 'llm', :text)
                    RETURNING id
                """),
                {"article_id": article_id, "text": summary_text},
            )
            summary_id = summary_result.scalar_one()

            # 4. Run (mit run_id als String in config JSONB)
            config_json = json.dumps({"proof_run_id": run_id})
            run_result = conn.execute(
                text("""
                    INSERT INTO runs (article_id, summary_id, run_type, status, config)
                    VALUES (:article_id, :summary_id, 'verification', 'success', CAST(:config AS jsonb))
                    RETURNING id
                """),
                {
                    "article_id": article_id,
                    "summary_id": summary_id,
                    "config": config_json,
                },
            )
            pg_run_id = run_result.scalar_one()

            # 5. Explainability Report (erstelle Tabelle falls nicht vorhanden)
            report_json_str = json.dumps(explainability.model_dump(mode="json"))

            # Erstelle Tabelle falls nicht vorhanden (mit IF NOT EXISTS)
            # Verwende SAVEPOINT für robuste Fehlerbehandlung
            conn.execute(text("SAVEPOINT before_table_setup"))
            try:
                # Tabelle erstellen (IF NOT EXISTS verhindert Fehler wenn bereits vorhanden)
                conn.execute(
                    text("""
                        CREATE TABLE IF NOT EXISTS explainability_reports (
                            id          SERIAL PRIMARY KEY,
                            run_id      INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                            version     TEXT NOT NULL,
                            report_json JSONB NOT NULL,
                            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                        )
                    """),
                )

                # Constraint hinzufügen (nur wenn nicht vorhanden)
                # Prüfe zuerst ob Constraint existiert
                constraint_check = conn.execute(
                    text("""
                        SELECT EXISTS (
                            SELECT 1 FROM pg_constraint 
                            WHERE conname = 'uq_explainability_reports_run_version'
                        )
                    """),
                )
                constraint_exists = constraint_check.scalar_one()

                if not constraint_exists:
                    conn.execute(
                        text("""
                            ALTER TABLE explainability_reports
                            ADD CONSTRAINT uq_explainability_reports_run_version 
                            UNIQUE (run_id, version)
                        """),
                    )

                # Index erstellen (IF NOT EXISTS verhindert Fehler wenn bereits vorhanden)
                conn.execute(
                    text("""
                        CREATE INDEX IF NOT EXISTS idx_explainability_reports_run_id
                        ON explainability_reports(run_id)
                    """),
                )

                conn.execute(text("RELEASE SAVEPOINT before_table_setup"))
            except Exception as e:
                # Bei Fehler: Rollback zu SAVEPOINT und prüfe ob Tabelle trotzdem existiert
                conn.execute(text("ROLLBACK TO SAVEPOINT before_table_setup"))
                # Prüfe ob Tabelle existiert (vielleicht wurde sie parallel erstellt)
                check_result = conn.execute(
                    text("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_schema = 'public' 
                            AND table_name = 'explainability_reports'
                        )
                    """),
                )
                if not check_result.scalar_one():
                    raise Exception(
                        f"Tabelle explainability_reports existiert nicht und konnte nicht erstellt werden: {e}"
                    )

            # Jetzt INSERT ausführen
            conn.execute(
                text("""
                    INSERT INTO explainability_reports (run_id, version, report_json)
                    VALUES (:run_id, :version, CAST(:report_json AS jsonb))
                    ON CONFLICT (run_id, version) DO UPDATE
                    SET report_json = EXCLUDED.report_json
                """),
                {
                    "run_id": pg_run_id,
                    "version": explainability.version,
                    "report_json": report_json_str,
                },
            )

            conn.commit()
            return True, pg_run_id, None
    except Exception as e:
        error_msg = str(e)
        print(f"⚠️  Postgres Store fehlgeschlagen: {error_msg}", file=sys.stderr)
        return False, None, error_msg


def store_proof_run_neo4j(
    driver,
    run_id: str,
    article_id: int,
    summary_id: int,
    explainability: ExplainabilityResult,
) -> tuple[bool, str | None]:
    """
    Speichert Proof-Run in Neo4j.
    Returns: (success, error_message)
    """
    if not driver:
        return False, "Neo4j driver nicht verfügbar"

    try:
        with driver.session() as session:
            session.execute_write(
                _store_explainability_neo4j_tx,
                run_id,
                article_id,
                summary_id,
                explainability,
            )
        return True, None
    except Exception as e:
        error_msg = str(e)
        print(f"⚠️  Neo4j Store fehlgeschlagen: {error_msg}", file=sys.stderr)
        return False, error_msg


def _store_explainability_neo4j_tx(
    tx, run_id: str, article_id: int, summary_id: int, explainability: ExplainabilityResult
):
    """Transaction für Neo4j Explainability-Persistenz."""
    # Run-Node mit run_id Property
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

    # Explainability-Node
    tx.run(
        """
        MATCH (r:Run {run_id: $run_id})
        MERGE (e:Explainability {run_id: $run_id, version: $version})
        SET e.summary = $summary
        MERGE (r)-[:HAS_EXPLAINABILITY]->(e)
        """,
        run_id=run_id,
        version=explainability.version,
        summary=json.dumps(explainability.summary),
    )

    # Findings
    for finding in explainability.findings:
        tx.run(
            """
            MATCH (e:Explainability {run_id: $run_id})
            MERGE (f:Finding {id: $finding_id, run_id: $run_id})
            SET f.dimension = $dimension,
                f.severity = $severity,
                f.message = $message
            MERGE (e)-[:HAS_FINDING]->(f)
            """,
            run_id=run_id,
            finding_id=finding.id,
            dimension=finding.dimension.value,
            severity=finding.severity,
            message=finding.message,
        )

        # Span (wenn vorhanden)
        if finding.span:
            tx.run(
                """
                MATCH (f:Finding {id: $finding_id, run_id: $run_id})
                MERGE (sp:Span {
                    run_id: $run_id,
                    finding_id: $finding_id,
                    start_char: $start_char,
                    end_char: $end_char
                })
                SET sp.text = $text
                MERGE (f)-[:HAS_SPAN]->(sp)
                """,
                run_id=run_id,
                finding_id=finding.id,
                start_char=finding.span.start_char,
                end_char=finding.span.end_char,
                text=finding.span.text or "",
            )


def verify_postgres(engine: Engine, run_id: str) -> dict:
    """Verifiziert Postgres-Einträge für Proof-Run."""
    checks = {
        "run_exists": False,
        "explainability_report_exists": False,
        "run_id_in_config": False,
        "report_json_valid": False,
    }

    try:
        with engine.connect() as conn:
            # Prüfe, ob Run mit proof_run_id existiert
            result = conn.execute(
                text("""
                    SELECT r.id, r.config
                    FROM runs r
                    WHERE r.config->>'proof_run_id' = :run_id
                    LIMIT 1
                """),
                {"run_id": run_id},
            )
            row = result.first()
            if row:
                checks["run_exists"] = True
                config = row[1]
                if config and config.get("proof_run_id") == run_id:
                    checks["run_id_in_config"] = True
                pg_run_id = row[0]

                # Prüfe Explainability Report
                # Zuerst Count prüfen
                result = conn.execute(
                    text("""
                        SELECT COUNT(*) as cnt
                        FROM explainability_reports
                        WHERE run_id = :run_id
                    """),
                    {"run_id": pg_run_id},
                )
                count = result.scalar_one()
                checks["explainability_report_exists"] = count > 0

                # Dann report_json separat abfragen und validieren
                if count > 0:
                    result = conn.execute(
                        text("""
                            SELECT report_json
                            FROM explainability_reports
                            WHERE run_id = :run_id
                            LIMIT 1
                        """),
                        {"run_id": pg_run_id},
                    )
                    row = result.first()
                    if row and row[0]:
                        try:
                            report_data = row[0]
                            if isinstance(report_data, dict) and "findings" in report_data:
                                checks["report_json_valid"] = True
                        except Exception:
                            pass
    except Exception as e:
        print(f"⚠️  Postgres Verify fehlgeschlagen: {e}", file=sys.stderr)

    return checks


def verify_neo4j(driver, run_id: str) -> dict:
    """Verifiziert Neo4j-Einträge für Proof-Run."""
    checks = {
        "run_node_exists": False,
        "run_has_run_id_property": False,
        "explainability_node_exists": False,
        "findings_exist": False,
        "spans_exist": False,
    }

    if not driver:
        return checks

    try:
        with driver.session() as session:
            # Run-Node prüfen
            result = session.run(
                """
                MATCH (r:Run)
                WHERE r.id = $run_id OR r.run_id = $run_id
                RETURN r.id as id, r.run_id as run_id_prop
                LIMIT 1
            """,
                run_id=run_id,
            )
            record = result.single()
            if record:
                checks["run_node_exists"] = True
                if record["run_id_prop"]:
                    checks["run_has_run_id_property"] = True

            # Explainability-Node prüfen
            result = session.run(
                """
                MATCH (e:Explainability {run_id: $run_id})
                RETURN count(e) as cnt
            """,
                run_id=run_id,
            )
            count = result.single()["cnt"]
            checks["explainability_node_exists"] = count > 0

            # Findings prüfen
            result = session.run(
                """
                MATCH (f:Finding {run_id: $run_id})
                RETURN count(f) as cnt
            """,
                run_id=run_id,
            )
            count = result.single()["cnt"]
            checks["findings_exist"] = count > 0

            # Spans prüfen
            result = session.run(
                """
                MATCH (sp:Span {run_id: $run_id})
                RETURN count(sp) as cnt
            """,
                run_id=run_id,
            )
            count = result.single()["cnt"]
            checks["spans_exist"] = count > 0
    except Exception as e:
        print(f"⚠️  Neo4j Verify fehlgeschlagen: {e}", file=sys.stderr)

    return checks


def check_cross_store_match(
    engine: Engine,
    driver,
    run_id: str,
) -> dict:
    """Prüft, ob run_id in beiden Stores existiert (Cross-Store Match)."""
    checks = {
        "postgres_has_run_id": False,
        "neo4j_has_run_id": False,
        "cross_store_match": False,
    }

    # Postgres: Prüfe ob Run mit proof_run_id existiert
    if engine:
        try:
            with engine.connect() as conn:
                result = conn.execute(
                    text("""
                        SELECT COUNT(*) as cnt
                        FROM runs r
                        WHERE r.config->>'proof_run_id' = :run_id
                    """),
                    {"run_id": run_id},
                )
                count = result.scalar_one()
                checks["postgres_has_run_id"] = count > 0
        except Exception as e:
            print(f"⚠️  Cross-Store Check Postgres fehlgeschlagen: {e}", file=sys.stderr)

    # Neo4j: Prüfe ob Run-Node mit run_id existiert
    if driver:
        try:
            with driver.session() as session:
                result = session.run(
                    """
                    MATCH (r:Run)
                    WHERE r.id = $run_id OR r.run_id = $run_id
                    RETURN count(r) as cnt
                """,
                    run_id=run_id,
                )
                count = result.single()["cnt"]
                checks["neo4j_has_run_id"] = count > 0
        except Exception as e:
            print(f"⚠️  Cross-Store Check Neo4j fehlgeschlagen: {e}", file=sys.stderr)

    # Match: beide Stores haben die run_id
    checks["cross_store_match"] = checks["postgres_has_run_id"] and checks["neo4j_has_run_id"]

    return checks


def write_proof_report(
    run_id: str,
    postgres_saved: bool,
    postgres_error: str | None,
    neo4j_saved: bool,
    neo4j_error: str | None,
    postgres_checks: dict,
    neo4j_checks: dict,
    cross_store_checks: dict,
    out_path: Path,
) -> None:
    """Schreibt Proof-Report als Markdown."""
    lines = [
        "# Explainability-Persistenz: Proof",
        "",
        f"**Datum:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Run-ID:** {run_id}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
    ]

    # Overall Status
    pg_pass = postgres_saved and all(postgres_checks.values()) if postgres_checks else False
    neo4j_pass = neo4j_saved and all(neo4j_checks.values()) if neo4j_checks else False
    cross_store_pass = (
        cross_store_checks.get("cross_store_match", False) if cross_store_checks else False
    )

    overall_pass = pg_pass and neo4j_pass and cross_store_pass

    lines.append(f"- **Postgres:** {'✅ PASS' if pg_pass else '❌ FAIL'}")
    if postgres_error:
        lines.append(f"  - Fehler: {postgres_error}")

    lines.append(f"- **Neo4j:** {'✅ PASS' if neo4j_pass else '❌ FAIL'}")
    if neo4j_error:
        lines.append(f"  - Fehler: {neo4j_error}")

    lines.append(f"- **Cross-Store Match:** {'✅ PASS' if cross_store_pass else '❌ FAIL'}")

    lines.append("")
    lines.append(f"- **Overall:** {'✅ PASS' if overall_pass else '❌ FAIL'}")

    lines.extend(
        [
            "",
            "---",
            "",
            "## Postgres Verification",
            "",
            f"**Gespeichert:** {'✅' if postgres_saved else '❌'}",
            "",
            "| Check | Status |",
            "|-------|--------|",
        ]
    )

    for check, passed in postgres_checks.items():
        status = "✅" if passed else "❌"
        lines.append(f"| {check} | {status} |")

    if neo4j_checks:
        lines.extend(
            [
                "",
                "## Neo4j Verification",
                "",
                f"**Gespeichert:** {'✅' if neo4j_saved else '❌'}",
                "",
                "| Check | Status |",
                "|-------|--------|",
            ]
        )

        for check, passed in neo4j_checks.items():
            status = "✅" if passed else "❌"
            lines.append(f"| {check} | {status} |")

    if cross_store_checks:
        lines.extend(
            [
                "",
                "## Cross-Store Consistency",
                "",
                "| Check | Status |",
                "|-------|--------|",
            ]
        )

        for check, passed in cross_store_checks.items():
            status = "✅" if passed else "❌"
            lines.append(f"| {check} | {status} |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## Evidence",
            "",
            "### Postgres Queries",
            "",
            "```sql",
            "-- Run mit proof_run_id finden",
            f"SELECT r.id, r.config FROM runs r WHERE r.config->>'proof_run_id' = '{run_id}';",
            "",
            "-- Explainability Report finden",
            "SELECT er.id, er.version, er.created_at",
            "FROM explainability_reports er",
            "JOIN runs r ON er.run_id = r.id",
            f"WHERE r.config->>'proof_run_id' = '{run_id}';",
            "```",
            "",
            "### Neo4j Queries",
            "",
            "```cypher",
            "-- Run-Node finden",
            f"MATCH (r:Run) WHERE r.id = '{run_id}' OR r.run_id = '{run_id}' RETURN r;",
            "",
            "-- Explainability-Node finden",
            f"MATCH (e:Explainability {{run_id: '{run_id}'}}) RETURN e;",
            "",
            "-- Findings finden",
            f"MATCH (f:Finding {{run_id: '{run_id}'}}) RETURN count(f) as findings_count;",
            "```",
            "",
            "---",
            "",
            "## Reproduktion",
            "",
            "### Command",
            "",
            "```bash",
            "python scripts/prove_explainability_persistence.py",
            f"  --run-id {run_id}",
            "  --fixture minimal",
            "  --out docs/status/explainability_persistence_proof.md",
            "```",
            "",
            "### Erwartete PASS-Kriterien",
            "",
            "1. **Postgres:**",
            "   - Run mit `proof_run_id` in `config` JSONB gespeichert",
            "   - Explainability Report in `explainability_reports` Tabelle vorhanden",
            "   - Report JSON ist gültig und enthält `findings`",
            "",
            "2. **Neo4j:**",
            "   - Run-Node mit `run_id` Property existiert",
            "   - Explainability-Node mit `run_id` existiert",
            "   - Mindestens 1 Finding-Node vorhanden",
            "   - Mindestens 1 Span-Node vorhanden (falls Findings Spans haben)",
            "",
            "3. **Cross-Store:**",
            "   - Dieselbe `run_id` existiert in Postgres (via `config->>'proof_run_id'`)",
            "   - Dieselbe `run_id` existiert in Neo4j (via `Run.run_id` Property)",
            "",
            "### Environment Variables",
            "",
            "**Postgres:**",
            "- `POSTGRES_DSN` oder",
            "- `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`",
            "",
            "**Neo4j:**",
            "- `NEO4J_URI` (z.B. `bolt://localhost:7687`)",
            "- `NEO4J_USER` (default: `neo4j`)",
            "- `NEO4J_PASSWORD`",
            "",
            "**Hinweis:** Falls eine DB nicht verfügbar ist, verwende `--skip-postgres` oder `--skip-neo4j`.",
            "",
        ]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ Proof-Report geschrieben: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Explainability-Persistenz Proof")
    ap.add_argument(
        "--run-id",
        type=str,
        default=f"proof-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Run-ID für Proof (default: proof-<timestamp>)",
    )
    ap.add_argument(
        "--fixture",
        type=str,
        default="minimal",
        choices=["minimal", "mixed", "edgecases"],
        help="Fixture-Name (default: minimal)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="docs/status/explainability_persistence_proof.md",
        help="Output-Pfad (default: docs/status/explainability_persistence_proof.md)",
    )
    ap.add_argument(
        "--skip-postgres",
        action="store_true",
        help="Postgres-Persistenz überspringen",
    )
    ap.add_argument(
        "--skip-neo4j",
        action="store_true",
        help="Neo4j-Persistenz überspringen",
    )

    args = ap.parse_args()

    print(f"🔍 Explainability-Persistenz Proof: {args.run_id}")
    print("")

    # Load fixture
    fixture_data = load_fixture(args.fixture)
    summary_text = fixture_data["summary_text"]
    article_text = summary_text  # Für Proof: verwende Summary als Article (vereinfacht)

    # Build Explainability (ohne LLM)
    print("📊 Explainability bauen...")
    service = ExplainabilityService()
    explainability = service.build(fixture_data, summary_text)
    print(
        f"   ✅ {len(explainability.findings)} Findings, {len(explainability.top_spans)} Top-Spans"
    )
    print("")

    # Postgres
    postgres_saved = False
    postgres_error = None
    postgres_checks = {}
    engine = None

    if not args.skip_postgres:
        print("💾 Postgres speichern...")
        engine = get_postgres_engine()
        if engine:
            success, pg_run_id, error = store_proof_run_postgres(
                engine,
                args.run_id,
                article_text,
                summary_text,
                explainability,
            )
            if success:
                print("   ✅ Gespeichert")
                postgres_saved = True
                postgres_checks = verify_postgres(engine, args.run_id)
                print(
                    f"   ✅ Verifiziert: {sum(postgres_checks.values())}/{len(postgres_checks)} Checks"
                )
            else:
                print(f"   ❌ Fehlgeschlagen: {error}")
                postgres_error = error
        else:
            print("   ⚠️  Postgres nicht verfügbar")
            postgres_error = "Postgres engine nicht verfügbar"
    else:
        print("   ⏭️  Übersprungen (--skip-postgres)")

    # Neo4j
    neo4j_saved = False
    neo4j_error = None
    neo4j_checks = {}
    driver = None

    if not args.skip_neo4j:
        print("💾 Neo4j speichern...")
        driver = get_neo4j_driver()
        if driver:
            # Für Neo4j brauchen wir article_id und summary_id (aus Postgres oder Mock)
            article_id = 999999  # Mock-ID für Proof
            summary_id = 999999
            success, error = store_proof_run_neo4j(
                driver,
                args.run_id,
                article_id,
                summary_id,
                explainability,
            )
            if success:
                print("   ✅ Gespeichert")
                neo4j_saved = True
                neo4j_checks = verify_neo4j(driver, args.run_id)
                print(f"   ✅ Verifiziert: {sum(neo4j_checks.values())}/{len(neo4j_checks)} Checks")
            else:
                print(f"   ❌ Fehlgeschlagen: {error}")
                neo4j_error = error
        else:
            print("   ⚠️  Neo4j nicht verfügbar")
            neo4j_error = "Neo4j driver nicht verfügbar"
    else:
        print("   ⏭️  Übersprungen (--skip-neo4j)")

    # Cross-Store Match Check
    print("")
    print("🔗 Cross-Store Match prüfen...")
    cross_store_checks = {}
    if engine and driver:
        cross_store_checks = check_cross_store_match(engine, driver, args.run_id)
        match_count = sum(cross_store_checks.values())
        print(f"   ✅ {match_count}/{len(cross_store_checks)} Checks")
    else:
        print("   ⚠️  Cross-Store Check übersprungen (Postgres oder Neo4j nicht verfügbar)")

    # Report
    print("")
    print("📝 Report generieren...")
    out_path = Path(args.out)
    write_proof_report(
        args.run_id,
        postgres_saved,
        postgres_error,
        neo4j_saved,
        neo4j_error,
        postgres_checks,
        neo4j_checks,
        cross_store_checks,
        out_path,
    )

    # Exit code: PASS nur wenn beide Stores gespeichert UND verifiziert UND run_id matchen
    pg_pass = postgres_saved and all(postgres_checks.values()) if postgres_checks else False
    neo4j_pass = neo4j_saved and all(neo4j_checks.values()) if neo4j_checks else False
    cross_store_pass = (
        cross_store_checks.get("cross_store_match", False) if cross_store_checks else False
    )

    overall_pass = pg_pass and neo4j_pass and cross_store_pass

    print("")
    if overall_pass:
        print("✅ Proof abgeschlossen: PASS")
        sys.exit(0)
    else:
        print("❌ Proof abgeschlossen: FAIL")
        if not pg_pass:
            print(
                f"   Postgres: {'nicht gespeichert' if not postgres_saved else 'Verifikation fehlgeschlagen'}"
            )
        if not neo4j_pass:
            print(
                f"   Neo4j: {'nicht gespeichert' if not neo4j_saved else 'Verifikation fehlgeschlagen'}"
            )
        if not cross_store_pass:
            print("   Cross-Store: run_id Match fehlgeschlagen")
        sys.exit(1)


if __name__ == "__main__":
    main()
