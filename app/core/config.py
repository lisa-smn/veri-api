from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # .env wird automatisch gelesen
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "VerificationAPI"
    environment: str = "dev"

    # Default: Docker-Host "db"
    # Lokal: wird durch .env / Env-Var DATABASE_URL überschrieben
    database_url: str = "postgresql+psycopg://veri:veri@db:5432/veri_db"

    neo4j_url: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "changeme"

    llm_url: str = "http://localhost:9000"


settings = Settings()
