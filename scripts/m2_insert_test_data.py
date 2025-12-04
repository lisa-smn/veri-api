from datetime import datetime
import json

from sqlalchemy import text
from app.db.session import SessionLocal


def main():
    db = SessionLocal()
    try:
        # legt ein Test-Dataset an
        dataset_id = db.execute(
            text("""
                INSERT INTO datasets (name, split, description)
                VALUES (:name, :split, :desc)
                RETURNING id
            """),
            {
                "name": "FineSumFact",
                "split": "test",
                "desc": "M2 Test-Dataset"
            }
        ).scalar_one()

        # Test-Artikel einfügen
        article_id = db.execute(
            text("""
                INSERT INTO articles (dataset_id, external_id, title, text)
                VALUES (:ds_id, :ext_id, :title, :text)
                RETURNING id
            """),
            {
                "ds_id": dataset_id,
                "ext_id": "article-001",
                "title": "Dummy-Artikel für M2",
                "text": "Dies ist ein Beispielartikel, um M2 zu testen."
            }
        ).scalar_one()

        # Dummy-Summary anlegen
        summary_id = db.execute(
            text("""
                INSERT INTO summaries (article_id, source, text, llm_model, prompt_version)
                VALUES (:article_id, :source, :text, :llm_model, :prompt_version)
                RETURNING id
            """),
            {
                "article_id": article_id,
                "source": "llm",
                "text": "Kurzfassung des Beispielartikels.",
                "llm_model": "gpt-4.1-mini",
                "prompt_version": "m2-test-v1"
            }
        ).scalar_one()

        # ein Fake-Run für M2
        run_id = db.execute(
            text("""
                INSERT INTO runs (article_id, summary_id, run_type, config, status, started_at, finished_at)
                VALUES (:article_id, :summary_id, :run_type, :config, :status, :started_at, :finished_at)
                RETURNING id
            """),
            {
                "article_id": article_id,
                "summary_id": summary_id,
                "run_type": "verification",
                "config": json.dumps({"agents": ["fact_agent"], "note": "M2 Fake Run"}),
                "status": "success",
                "started_at": datetime.utcnow(),
                "finished_at": datetime.utcnow(),
            }
        ).scalar_one()

        # ein Fake-Result dazu
        db.execute(
            text("""
                INSERT INTO verification_results (run_id, dimension, score, label, details)
                VALUES (:run_id, :dimension, :score, :label, :details)
            """),
            {
                "run_id": run_id,
                "dimension": "factuality",
                "score": 0.9,
                "label": "pass",
                "details": json.dumps({"note": "Fake-Result für M2"}),
            }
        )

        db.commit()
        print(
            f"M2 Testdaten angelegt: dataset={dataset_id}, "
            f"article={article_id}, summary={summary_id}, run={run_id}"
        )

    finally:
        db.close()


main()
