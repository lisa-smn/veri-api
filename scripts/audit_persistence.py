#!/usr/bin/env python3
"""
Persistence Audit für Postgres + Neo4j.

Prüft:
- Postgres: Tabellen-Inventar, Integrity (Duplicates, Missingness), Constraints
- Neo4j: Labels/Relations, Constraints/Indexes, Dangling Nodes, Duplicates
- Cross-Store: Konsistenz zwischen Postgres und Neo4j (run_id overlap)

Output: docs/status/persistence_audit.md
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import sys
from typing import Any

try:
    from sqlalchemy import create_engine, inspect, text
    from sqlalchemy.engine import Engine
except ImportError:
    print(
        "❌ FEHLER: sqlalchemy nicht installiert. Installiere: pip install sqlalchemy",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from neo4j import GraphDatabase
except ImportError:
    print("❌ FEHLER: neo4j nicht installiert. Installiere: pip install neo4j", file=sys.stderr)
    sys.exit(1)

try:
    from dotenv import load_dotenv

    # Load .env (optional, falls vorhanden)
    # override=False: ENV-Variablen (export) haben Priorität über .env
    try:
        load_dotenv(override=False)
    except (PermissionError, FileNotFoundError):
        # .env nicht verfügbar (z.B. in Sandbox), verwende ENV-Variablen direkt
        pass
except ImportError:
    # python-dotenv nicht installiert, verwende ENV-Variablen direkt
    pass

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from app.core.config import settings
except (ImportError, PermissionError, FileNotFoundError) as e:
    # Import kann fehlschlagen, wenn .env nicht lesbar ist (Sandbox) oder Module fehlen
    print(
        f"⚠️  WARNING: app.core.config nicht importierbar ({type(e).__name__}), verwende ENV-Variablen",
        file=sys.stderr,
    )
    settings = None


@dataclass
class PostgresFindings:
    """Postgres Audit-Ergebnisse."""

    tables: list[dict[str, Any]]
    run_tables: list[str]
    duplicates: dict[str, int]
    missingness: dict[str, dict[str, int]]
    constraints: dict[str, list[str]]
    sample_rows: dict[str, list[dict[str, Any]]]
    errors: list[str]


@dataclass
class Neo4jFindings:
    """Neo4j Audit-Ergebnisse."""

    labels: list[str]
    rel_types: list[str]
    label_counts: dict[str, int]
    constraints: list[dict[str, Any]]
    indexes: list[dict[str, Any]]
    run_nodes: list[str]
    dangling_nodes: dict[str, int]
    duplicates: dict[str, int]
    errors: list[str]


@dataclass
class CrossStoreFindings:
    """Cross-Store Konsistenz-Ergebnisse."""

    postgres_run_ids: list[str]
    neo4j_run_ids: list[str]
    only_in_postgres: list[str]
    only_in_neo4j: list[str]
    overlap_count: int
    sample_join_checks: list[dict[str, Any]]
    errors: list[str]


def get_postgres_engine() -> Engine | None:
    """Erstellt Postgres Engine aus Config oder ENV."""
    if settings:
        db_url = settings.database_url
    else:
        # Fallback: ENV-Variablen
        host = os.getenv("POSTGRES_HOST") or os.getenv("DATABASE_HOST") or "localhost"
        port = os.getenv("POSTGRES_PORT") or os.getenv("DATABASE_PORT") or "5432"
        db = os.getenv("POSTGRES_DB") or os.getenv("DATABASE_NAME") or "veri_db"
        user = os.getenv("POSTGRES_USER") or os.getenv("DATABASE_USER") or "veri"
        password = os.getenv("POSTGRES_PASSWORD") or os.getenv("DATABASE_PASSWORD") or "veri"
        db_url = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}"

    try:
        engine = create_engine(db_url, echo=False)
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        print(f"⚠️  WARNING: Postgres-Verbindung fehlgeschlagen: {e}", file=sys.stderr)
        return None


def resolve_neo4j_config():
    """
    Resolved Neo4j-Konfiguration mit korrekter Priorität:
    1. ENV-Variablen (höchste Priorität)
    2. settings aus app.core.config (Fallback)

    Returns: (uri, user, password, source)
    """
    # Priorität 1: ENV-Variablen (mehrere Alias-Namen unterstützen)
    env_uri = os.getenv("NEO4J_URI") or os.getenv("NEO4J_URL") or os.getenv("NEO4J_BOLT_URL")
    env_user = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")
    env_password = os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASS")

    if env_uri:
        # ENV hat Priorität
        uri = env_uri
        user = env_user or (settings.neo4j_user if settings else "neo4j")
        password = env_password or (settings.neo4j_password if settings else "changeme")
        source = "env"
    elif settings:
        # Fallback: settings aus app.core.config
        uri = settings.neo4j_url
        user = settings.neo4j_user
        password = settings.neo4j_password
        source = "settings_default"
    else:
        # Letzter Fallback: Hardcoded defaults
        uri = "bolt://localhost:7687"
        user = "neo4j"
        password = "changeme"
        source = "hardcoded_default"

    return uri, user, password, source


def get_neo4j_driver():
    """Erstellt Neo4j Driver aus Config oder ENV."""
    uri, user, password, source = resolve_neo4j_config()

    # Debug-Ausgabe
    print(f"🔗 Neo4j target: {uri} (source={source})", file=sys.stderr)
    if user:
        print(f"   User: {user}", file=sys.stderr)

    # Optional: Host-vs-Docker Hint (nur wenn URI "neo4j:" als Host enthält, nicht "localhost")
    if "://neo4j:" in uri or (uri.startswith("bolt://neo4j") and "localhost" not in uri):
        try:
            import socket

            # Versuche DNS-Resolution (nur als Hint, nicht als Fehler)
            socket.gethostbyname("neo4j")
        except (socket.gaierror, OSError):
            print(
                "   ⚠️  Hint: 'neo4j' host not resolvable. Running on host? Use: export NEO4J_URI=bolt://localhost:7687",
                file=sys.stderr,
            )

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        # Test connection
        with driver.session() as session:
            session.run("RETURN 1")
        return driver
    except Exception as e:
        print(f"⚠️  WARNING: Neo4j-Verbindung fehlgeschlagen: {e}", file=sys.stderr)
        return None


def audit_postgres(engine: Engine) -> PostgresFindings:
    """Auditiert Postgres-Datenbank."""
    findings = PostgresFindings(
        tables=[],
        run_tables=[],
        duplicates={},
        missingness={},
        constraints={},
        sample_rows={},
        errors=[],
    )

    try:
        inspector = inspect(engine)
        all_tables = inspector.get_table_names()

        # Table inventory + row counts
        for table_name in all_tables:
            try:
                with engine.connect() as conn:
                    # Row count
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                    row_count = result.scalar()

                    # Columns
                    columns = [col["name"] for col in inspector.get_columns(table_name)]

                    findings.tables.append(
                        {
                            "name": table_name,
                            "row_count": row_count,
                            "columns": columns,
                        }
                    )

                    # Check if run-related
                    if any(
                        keyword in table_name.lower()
                        for keyword in ["run", "evaluation", "metric", "result"]
                    ):
                        findings.run_tables.append(table_name)

                    # Missingness checks (key columns)
                    key_columns = ["run_id", "example_id", "created_at", "dimension", "score"]
                    missingness = {}
                    for col in key_columns:
                        if col in columns:
                            result = conn.execute(
                                text(f"SELECT COUNT(*) FROM {table_name} WHERE {col} IS NULL")
                            )
                            null_count = result.scalar()
                            if null_count > 0:
                                missingness[col] = null_count
                    if missingness:
                        findings.missingness[table_name] = missingness

                    # Duplicate checks
                    if "run_id" in columns:
                        result = conn.execute(
                            text(
                                f"SELECT run_id, COUNT(*) as cnt FROM {table_name} GROUP BY run_id HAVING COUNT(*) > 1 LIMIT 10"
                            )
                        )
                        dupes = result.fetchall()
                        if dupes:
                            findings.duplicates[table_name] = len(dupes)

                    # Unique constraint on (run_id, dimension) if both exist
                    if "run_id" in columns and "dimension" in columns:
                        try:
                            result = conn.execute(
                                text(
                                    f"SELECT run_id, dimension, COUNT(*) as cnt FROM {table_name} WHERE run_id IS NOT NULL AND dimension IS NOT NULL GROUP BY run_id, dimension HAVING COUNT(*) > 1 LIMIT 10"
                                )
                            )
                            dupes = result.fetchall()
                            if dupes:
                                findings.duplicates[f"{table_name}.(run_id,dimension)"] = len(dupes)
                        except Exception as e:
                            findings.errors.append(
                                f"Duplicate check {table_name}.(run_id,dimension): {e}"
                            )

                    # Sample rows (newest runs)
                    if "run_id" in columns or "id" in columns:
                        order_col = "run_id" if "run_id" in columns else "id"
                        result = conn.execute(
                            text(f"SELECT * FROM {table_name} ORDER BY {order_col} DESC LIMIT 3")
                        )
                        rows = result.fetchall()
                        if rows:
                            findings.sample_rows[table_name] = [
                                {col: val for col, val in zip(columns, row)} for row in rows
                            ]

            except Exception as e:
                findings.errors.append(f"{table_name}: {e}")

        # Constraints
        for table_name in all_tables:
            try:
                constraints = inspector.get_unique_constraints(table_name)
                if constraints:
                    findings.constraints[table_name] = [c["name"] for c in constraints]
            except Exception as e:
                findings.errors.append(f"Constraints {table_name}: {e}")

    except Exception as e:
        findings.errors.append(f"Postgres Audit: {e}")

    return findings


def audit_neo4j(driver) -> Neo4jFindings:
    """Auditiert Neo4j-Datenbank."""
    findings = Neo4jFindings(
        labels=[],
        rel_types=[],
        label_counts={},
        constraints=[],
        indexes=[],
        run_nodes=[],
        dangling_nodes={},
        duplicates={},
        errors=[],
    )

    try:
        with driver.session() as session:
            # Labels
            result = session.run("CALL db.labels()")
            findings.labels = [record["label"] for record in result]

            # Relationship types
            result = session.run("CALL db.relationshipTypes()")
            findings.rel_types = [record["relationshipType"] for record in result]

            # Count per label
            for label in findings.labels:
                try:
                    result = session.run(f"MATCH (n:{label}) RETURN count(n) as cnt")
                    count = result.single()["cnt"]
                    findings.label_counts[label] = count
                except Exception as e:
                    findings.errors.append(f"Count {label}: {e}")

            # Constraints
            try:
                result = session.run("SHOW CONSTRAINTS")
                findings.constraints = [dict(record) for record in result]
            except Exception as e:
                findings.errors.append(f"Constraints: {e}")

            # Indexes
            try:
                result = session.run("SHOW INDEXES")
                findings.indexes = [dict(record) for record in result]
            except Exception as e:
                findings.errors.append(f"Indexes: {e}")

            # Find nodes with run_id property (oder id property für Run-Nodes)
            try:
                # Prüfe run_id property (für Metric, IssueSpan, etc.)
                result = session.run("""
                    MATCH (n)
                    WHERE n.run_id IS NOT NULL
                    RETURN labels(n) as labels, count(*) as cnt
                    ORDER BY cnt DESC
                """)
                for record in result:
                    labels = record["labels"]
                    label_str = ":".join(labels) if labels else "Unknown"
                    findings.run_nodes.append(f"{label_str} (run_id)")

                # Prüfe auch Run-Nodes mit id property (könnte run_id sein)
                result = session.run("""
                    MATCH (r:Run)
                    WHERE r.id IS NOT NULL
                    RETURN count(*) as cnt
                """)
                run_count = result.single()["cnt"]
                if run_count > 0:
                    findings.run_nodes.append("Run (id)")
            except Exception as e:
                findings.errors.append(f"Run nodes: {e}")

            # Dangling nodes (nodes with run_id but no relationships)
            try:
                for label in findings.labels:
                    result = session.run(f"""
                        MATCH (n:{label})
                        WHERE n.run_id IS NOT NULL AND NOT (n)-[]-()
                        RETURN count(n) as cnt
                    """)
                    count = result.single()["cnt"]
                    if count > 0:
                        findings.dangling_nodes[label] = count
            except Exception as e:
                findings.errors.append(f"Dangling {label}: {e}")

            # Duplicates (same run_id nodes >1 per label)
            try:
                for label in findings.labels:
                    result = session.run(f"""
                        MATCH (n:{label})
                        WHERE n.run_id IS NOT NULL
                        WITH n.run_id as run_id, count(*) as cnt
                        WHERE cnt > 1
                        RETURN count(*) as dup_count
                    """)
                    dup_count = result.single()["dup_count"]
                    if dup_count > 0:
                        findings.duplicates[label] = dup_count
            except Exception as e:
                findings.errors.append(f"Duplicates {label}: {e}")

    except Exception as e:
        findings.errors.append(f"Neo4j Audit: {e}")

    return findings


def audit_cross_store(
    engine: Engine | None,
    driver,
    postgres_findings: PostgresFindings,
    neo4j_findings: Neo4jFindings,
) -> CrossStoreFindings:
    """Prüft Konsistenz zwischen Postgres und Neo4j."""
    findings = CrossStoreFindings(
        postgres_run_ids=[],
        neo4j_run_ids=[],
        only_in_postgres=[],
        only_in_neo4j=[],
        overlap_count=0,
        sample_join_checks=[],
        errors=[],
    )

    if not engine or not driver:
        findings.errors.append("Postgres oder Neo4j nicht verfügbar")
        return findings

    try:
        # Postgres run_ids
        with engine.connect() as conn:
            # Find run_id column in any table
            for table in postgres_findings.run_tables:
                try:
                    inspector = inspect(engine)
                    columns = [col["name"] for col in inspector.get_columns(table)]
                    if "run_id" in columns:
                        result = conn.execute(
                            text(
                                f"SELECT DISTINCT run_id FROM {table} WHERE run_id IS NOT NULL LIMIT 100"
                            )
                        )
                        run_ids = [str(row[0]) for row in result]
                        findings.postgres_run_ids.extend(run_ids)
                except Exception as e:
                    findings.errors.append(f"Postgres run_ids {table}: {e}")

        findings.postgres_run_ids = list(set(findings.postgres_run_ids))

        # Neo4j run_ids (aus run_id property ODER Run.id)
        with driver.session() as session:
            # run_id property (Metric, IssueSpan, etc.)
            result = session.run("""
                MATCH (n)
                WHERE n.run_id IS NOT NULL
                RETURN DISTINCT n.run_id as run_id
                LIMIT 100
            """)
            run_ids_from_property = [str(record["run_id"]) for record in result]

            # Run.id (könnte auch run_id sein)
            result = session.run("""
                MATCH (r:Run)
                WHERE r.id IS NOT NULL
                RETURN DISTINCT r.id as run_id
                LIMIT 100
            """)
            run_ids_from_run = [str(record["run_id"]) for record in result]

            findings.neo4j_run_ids = list(set(run_ids_from_property + run_ids_from_run))

        # Overlap
        postgres_set = set(findings.postgres_run_ids)
        neo4j_set = set(findings.neo4j_run_ids)
        findings.overlap_count = len(postgres_set & neo4j_set)
        findings.only_in_postgres = list(postgres_set - neo4j_set)[:20]
        findings.only_in_neo4j = list(neo4j_set - postgres_set)[:20]

        # Sample join checks (für 3 run_ids)
        sample_run_ids = list(postgres_set & neo4j_set)[:3]
        for run_id in sample_run_ids:
            check = {"run_id": run_id, "postgres": False, "neo4j": False}

            # Check Postgres
            try:
                with engine.connect() as conn:
                    for table in postgres_findings.run_tables:
                        inspector = inspect(engine)
                        columns = [col["name"] for col in inspector.get_columns(table)]
                        if "run_id" in columns:
                            result = conn.execute(
                                text(f"SELECT COUNT(*) FROM {table} WHERE run_id = :run_id"),
                                {"run_id": run_id},
                            )
                            if result.scalar() > 0:
                                check["postgres"] = True
                                break
            except Exception as e:
                findings.errors.append(f"Sample check Postgres {run_id}: {e}")

            # Check Neo4j (run_id property ODER Run.id)
            try:
                with driver.session() as session:
                    result = session.run(
                        """
                        MATCH (n)
                        WHERE (n.run_id = $run_id OR (n:Run AND n.id = $run_id))
                        RETURN count(n) as cnt
                    """,
                        run_id=run_id,
                    )
                    if result.single()["cnt"] > 0:
                        check["neo4j"] = True
            except Exception as e:
                findings.errors.append(f"Sample check Neo4j {run_id}: {e}")

            findings.sample_join_checks.append(check)

    except Exception as e:
        findings.errors.append(f"Cross-Store Audit: {e}")

    return findings


def write_report(
    postgres_findings: PostgresFindings,
    neo4j_findings: Neo4jFindings,
    cross_store_findings: CrossStoreFindings,
    out_path: Path,
) -> None:
    """Schreibt Audit-Report als Markdown."""
    lines = [
        "# Persistence Audit: Postgres + Neo4j",
        "",
        f"**Datum:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "**Zweck:** Automatische Prüfung der Datenintegrität und Konsistenz zwischen Postgres und Neo4j",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
    ]

    # PASS/FAIL Kategorien
    pg_status = (
        "✅ PASS"
        if not postgres_findings.errors and not postgres_findings.duplicates
        else "⚠️  WARNINGS"
    )
    neo4j_status = (
        "✅ PASS" if not neo4j_findings.errors and not neo4j_findings.duplicates else "⚠️  WARNINGS"
    )
    cross_status = (
        "✅ PASS"
        if cross_store_findings.overlap_count > 0 and not cross_store_findings.errors
        else "⚠️  WARNINGS"
    )

    lines.extend(
        [
            f"- **Postgres:** {pg_status}",
            f"- **Neo4j:** {neo4j_status}",
            f"- **Cross-Store Konsistenz:** {cross_status}",
            "",
            "---",
            "",
            "## Postgres Findings",
            "",
            "### Table Inventory",
            "",
            "| Table | Row Count | Key Columns |",
            "|-------|-----------|--------------|",
        ]
    )

    for table in postgres_findings.tables:
        key_cols = [
            col
            for col in ["run_id", "example_id", "created_at", "dimension", "score"]
            if col in table["columns"]
        ]
        key_cols_str = ", ".join(key_cols) if key_cols else "-"
        lines.append(f"| {table['name']} | {table['row_count']} | {key_cols_str} |")

    lines.extend(
        [
            "",
            f"**Run-related Tables:** {', '.join(postgres_findings.run_tables) if postgres_findings.run_tables else 'Keine gefunden'}",
            "",
            "### Constraints",
            "",
        ]
    )

    if postgres_findings.constraints:
        for table, constraints in postgres_findings.constraints.items():
            lines.append(f"- **{table}:** {', '.join(constraints)}")
    else:
        lines.append("- Keine Unique Constraints gefunden")

    lines.extend(
        [
            "",
            "### Duplicates",
            "",
        ]
    )

    if postgres_findings.duplicates:
        for key, count in postgres_findings.duplicates.items():
            lines.append(f"- **{key}:** {count} Duplikate gefunden ⚠️")
    else:
        lines.append("- ✅ Keine Duplikate gefunden")

    lines.extend(
        [
            "",
            "### Missingness (NULL in Key Columns)",
            "",
        ]
    )

    if postgres_findings.missingness:
        for table, missing in postgres_findings.missingness.items():
            missing_str = ", ".join([f"{col}={count}" for col, count in missing.items()])
            lines.append(f"- **{table}:** {missing_str} ⚠️")
    else:
        lines.append("- ✅ Keine NULL-Werte in Key Columns")

    lines.extend(
        [
            "",
            "### Sample Rows (Newest)",
            "",
        ]
    )

    for table, rows in list(postgres_findings.sample_rows.items())[:3]:
        lines.append(f"**{table}:**")
        for i, row in enumerate(rows[:2], 1):
            row_str = ", ".join([f"{k}={v}" for k, v in list(row.items())[:5]])
            lines.append(f"  {i}. {row_str}")
        lines.append("")

    if postgres_findings.errors:
        lines.extend(
            [
                "",
                "### Errors",
                "",
            ]
        )
        for error in postgres_findings.errors:
            lines.append(f"- ⚠️  {error}")

    lines.extend(
        [
            "",
            "---",
            "",
            "## Neo4j Findings",
            "",
            "### Labels & Relationship Types",
            "",
            f"**Labels:** {', '.join(neo4j_findings.labels) if neo4j_findings.labels else 'Keine gefunden'}",
            f"**Relationship Types:** {', '.join(neo4j_findings.rel_types) if neo4j_findings.rel_types else 'Keine gefunden'}",
            "",
            "### Label Counts",
            "",
            "| Label | Count |",
            "|-------|-------|",
        ]
    )

    for label, count in sorted(neo4j_findings.label_counts.items()):
        lines.append(f"| {label} | {count} |")

    lines.extend(
        [
            "",
            "### Constraints & Indexes",
            "",
            f"**Constraints:** {len(neo4j_findings.constraints)}",
            f"**Indexes:** {len(neo4j_findings.indexes)}",
            "",
            "### Run Nodes",
            "",
            f"**Labels mit run_id Property:** {', '.join(neo4j_findings.run_nodes) if neo4j_findings.run_nodes else 'Keine gefunden'}",
            "",
            "### Dangling Nodes (isolated, keine Relationships)",
            "",
        ]
    )

    if neo4j_findings.dangling_nodes:
        for label, count in neo4j_findings.dangling_nodes.items():
            lines.append(f"- **{label}:** {count} isolierte Nodes ⚠️")
    else:
        lines.append("- ✅ Keine isolierten Nodes gefunden")

    lines.extend(
        [
            "",
            "### Duplicates (same run_id >1 per label)",
            "",
        ]
    )

    if neo4j_findings.duplicates:
        for label, count in neo4j_findings.duplicates.items():
            lines.append(f"- **{label}:** {count} Duplikate gefunden ⚠️")
    else:
        lines.append("- ✅ Keine Duplikate gefunden")

    if neo4j_findings.errors:
        lines.extend(
            [
                "",
                "### Errors",
                "",
            ]
        )
        for error in neo4j_findings.errors:
            lines.append(f"- ⚠️  {error}")

    lines.extend(
        [
            "",
            "---",
            "",
            "## Cross-Store Consistency",
            "",
            f"**Postgres run_ids (distinct):** {len(cross_store_findings.postgres_run_ids)}",
            f"**Neo4j run_ids (distinct):** {len(cross_store_findings.neo4j_run_ids)}",
            f"**Overlap:** {cross_store_findings.overlap_count}",
            "",
        ]
    )

    if cross_store_findings.only_in_postgres:
        lines.extend(
            [
                "**Nur in Postgres (Sample, max 20):**",
                "",
            ]
        )
        for run_id in cross_store_findings.only_in_postgres[:10]:
            lines.append(f"- {run_id}")
        if len(cross_store_findings.only_in_postgres) > 10:
            lines.append(f"- ... und {len(cross_store_findings.only_in_postgres) - 10} weitere")
        lines.append("")

    if cross_store_findings.only_in_neo4j:
        lines.extend(
            [
                "**Nur in Neo4j (Sample, max 20):**",
                "",
            ]
        )
        for run_id in cross_store_findings.only_in_neo4j[:10]:
            lines.append(f"- {run_id}")
        if len(cross_store_findings.only_in_neo4j) > 10:
            lines.append(f"- ... und {len(cross_store_findings.only_in_neo4j) - 10} weitere")
        lines.append("")

    lines.extend(
        [
            "### Sample Join Checks",
            "",
            "| run_id | Postgres | Neo4j |",
            "|--------|----------|-------|",
        ]
    )

    for check in cross_store_findings.sample_join_checks:
        pg_str = "✅" if check["postgres"] else "❌"
        neo4j_str = "✅" if check["neo4j"] else "❌"
        lines.append(f"| {check['run_id']} | {pg_str} | {neo4j_str} |")

    if cross_store_findings.errors:
        lines.extend(
            [
                "",
                "### Errors",
                "",
            ]
        )
        for error in cross_store_findings.errors:
            lines.append(f"- ⚠️  {error}")

    lines.extend(
        [
            "",
            "---",
            "",
            "## Recommendations",
            "",
            "### P0 (Datenverlust / fehlende Verknüpfungen / Duplicates)",
            "",
        ]
    )

    p0_items = []
    if postgres_findings.duplicates:
        p0_items.append("- Postgres Duplikate beheben (siehe 'Duplicates' Abschnitt)")
    if neo4j_findings.duplicates:
        p0_items.append("- Neo4j Duplikate beheben (siehe 'Duplicates' Abschnitt)")
    if neo4j_findings.dangling_nodes:
        p0_items.append("- Isolierte Neo4j Nodes verknüpfen oder entfernen")
    if cross_store_findings.only_in_postgres or cross_store_findings.only_in_neo4j:
        p0_items.append("- Cross-Store Konsistenz prüfen: fehlende run_ids in einem Store")

    if p0_items:
        for item in p0_items:
            lines.append(item)
    else:
        lines.append("- ✅ Keine P0-Probleme gefunden")

    lines.extend(
        [
            "",
            "### P1 (Constraints/Indexes ergänzen)",
            "",
            "- Postgres: Unique Constraints für (run_id, dimension) prüfen",
            "- Neo4j: Constraints für run_id uniqueness pro Label prüfen",
            "- Indexes für häufige Queries prüfen (run_id, example_id)",
            "",
            "### P2 (Redundanz reduzieren, Archivierung)",
            "",
            "- Alte Runs archivieren (beide Stores)",
            "- Redundante Daten zwischen Postgres und Neo4j dokumentieren",
            "- Cleanup-Strategie für isolierte Nodes definieren",
            "",
        ]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ Report geschrieben: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Persistence Audit für Postgres + Neo4j")
    ap.add_argument(
        "--out",
        type=str,
        default="docs/status/persistence_audit.md",
        help="Output-Pfad für Report (default: docs/status/persistence_audit.md)",
    )

    args = ap.parse_args()

    out_path = Path(args.out)

    print("🔍 Persistence Audit gestartet...")
    print("")

    # Postgres
    print("📊 Postgres Audit...")
    engine = get_postgres_engine()
    if engine:
        postgres_findings = audit_postgres(engine)
        print(f"  ✅ {len(postgres_findings.tables)} Tabellen gefunden")
    else:
        print("  ⚠️  Postgres nicht verfügbar")
        postgres_findings = PostgresFindings([], [], {}, {}, {}, {}, ["Postgres nicht verfügbar"])

    # Neo4j
    print("📊 Neo4j Audit...")
    driver = get_neo4j_driver()
    if driver:
        neo4j_findings = audit_neo4j(driver)
        print(f"  ✅ {len(neo4j_findings.labels)} Labels gefunden")
    else:
        print("  ⚠️  Neo4j nicht verfügbar")
        neo4j_findings = Neo4jFindings([], [], {}, [], [], [], {}, {}, ["Neo4j nicht verfügbar"])

    # Cross-Store
    print("📊 Cross-Store Konsistenz...")
    cross_store_findings = audit_cross_store(engine, driver, postgres_findings, neo4j_findings)
    if cross_store_findings.overlap_count > 0:
        print(f"  ✅ {cross_store_findings.overlap_count} run_ids in beiden Stores")
    else:
        print("  ⚠️  Keine Overlap gefunden")

    # Report
    print("")
    print("📝 Report generieren...")
    write_report(postgres_findings, neo4j_findings, cross_store_findings, out_path)

    print("")
    print("✅ Audit abgeschlossen!")


if __name__ == "__main__":
    main()
