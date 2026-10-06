from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/healthcare_db"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "healthcare_db"
    POSTGRES_PORT: int = 5432

    OPENAI_API_KEY: Optional[str] = None
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gpt-4o-mini"

    SECRET_KEY: str = "dev-secret-key-change-in-production-12345"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    VECTOR_DB_PATH: str = os.path.join(os.path.dirname(__file__), "..", "..", "data", "vectorstore")
    MODEL_STORAGE_PATH: str = os.path.join(os.path.dirname(__file__), "..", "..", "data", "models")
    DATA_UPLOAD_PATH: str = os.path.join(os.path.dirname(__file__), "..", "..", "data", "uploads")

    MLFLOW_TRACKING_URI: str = os.path.join(os.path.dirname(__file__), "..", "..", "data", "mlruns")

    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


settings = Settings()
