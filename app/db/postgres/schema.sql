-- erstellt ENUM-Typen nur, wenn sie noch nicht existieren
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'summary_source') THEN
        CREATE TYPE summary_source AS ENUM ('reference', 'llm', 'baseline', 'other');
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'run_type') THEN
        CREATE TYPE run_type AS ENUM ('verification', 'metric_baseline');
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'run_status') THEN
        CREATE TYPE run_status AS ENUM ('pending', 'running', 'success', 'failed');
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'verification_dimension') THEN
        CREATE TYPE verification_dimension AS ENUM (
            'factuality',
            'coherence',
            'fluency',
            'readability',
            'overall'
        );
    END IF;
END$$;

-- speichert verwendete Datensätze
CREATE TABLE IF NOT EXISTS datasets (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    split       TEXT,
    description TEXT,
    source_url  TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Artikel aus den Datensätzen
CREATE TABLE IF NOT EXISTS articles (
    id              SERIAL PRIMARY KEY,
    dataset_id      INT REFERENCES datasets(id),
    external_id     TEXT,
    title           TEXT,
    text            TEXT NOT NULL,
    metadata        JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Zusammenfassungen zu Artikeln (LLM, Referenz, Baseline usw.)
CREATE TABLE IF NOT EXISTS summaries (
    id              SERIAL PRIMARY KEY,
    article_id      INT NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    source          summary_source NOT NULL,
    text            TEXT NOT NULL,
    llm_model       TEXT,
    prompt_version  TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ein Verifikationslauf pro Summary
CREATE TABLE IF NOT EXISTS runs (
    id              SERIAL PRIMARY KEY,
    article_id      INT NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    summary_id      INT NOT NULL REFERENCES summaries(id) ON DELETE CASCADE,
    run_type        run_type NOT NULL,
    config          JSONB,
    status          run_status NOT NULL DEFAULT 'pending',
    started_at      TIMESTAMPTZ DEFAULT NOW(),
    finished_at     TIMESTAMPTZ
);

-- Ergebnisse einer Verifikationsdimension (Fakten, Kohärenz usw.)
CREATE TABLE IF NOT EXISTS verification_results (
    id              SERIAL PRIMARY KEY,
    run_id          INT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    dimension       verification_dimension NOT NULL,
    score           DOUBLE PRECISION,
    label           TEXT,
    details         JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Fehler, die während eines Runs auftreten
CREATE TABLE IF NOT EXISTS errors (
    id              SERIAL PRIMARY KEY,
    run_id          INT REFERENCES runs(id) ON DELETE CASCADE,
    stage           TEXT,
    message         TEXT NOT NULL,
    traceback       TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Erklärungen der Agenten (Explainability)
CREATE TABLE IF NOT EXISTS explanations (
    id                      SERIAL PRIMARY KEY,
    run_id                  INT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    verification_result_id  INT REFERENCES verification_results(id) ON DELETE SET NULL,
    agent_name              TEXT,
    explanation             TEXT NOT NULL,
    raw_response            JSONB,
    created_at              TIMESTAMPTZ DEFAULT NOW()
);

-- sinnvolle Indexe für Performance
CREATE INDEX IF NOT EXISTS idx_articles_dataset_id
    ON articles (dataset_id);

CREATE INDEX IF NOT EXISTS idx_summaries_article_id
    ON summaries (article_id);

CREATE INDEX IF NOT EXISTS idx_runs_article_id
    ON runs (article_id);

CREATE INDEX IF NOT EXISTS idx_runs_summary_id
    ON runs (summary_id);

CREATE INDEX IF NOT EXISTS idx_runs_status
    ON runs (status);

CREATE INDEX IF NOT EXISTS idx_verification_results_run_id
    ON verification_results (run_id);
