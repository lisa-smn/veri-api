from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "VerificationAPI"      # Name der App
    environment: str = "dev"               # Umgebung (dev/prod)

    database_url: str = "postgresql+psycopg://veri:veri@db:5432/veri_db"  # DB-Connection

    neo4j_url: str = "bolt://localhost:7687"   # Neo4j-Adresse
    neo4j_user: str = "neo4j"                  # Neo4j-User
    neo4j_password: str = "changeme"           # Neo4j-Passwort

    llm_url: str = "http://localhost:9000"     # lokales LLM

    # liest Werte aus .env, falls vorhanden
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# globale Settings-Instanz
settings = Settings()
