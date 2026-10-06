from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from backend.config import settings

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
    os.makedirs(settings.MODEL_STORAGE_PATH, exist_ok=True)
    os.makedirs(settings.DATA_UPLOAD_PATH, exist_ok=True)
    os.makedirs(settings.MLFLOW_TRACKING_URI, exist_ok=True)
    Base.metadata.create_all(bind=engine)
