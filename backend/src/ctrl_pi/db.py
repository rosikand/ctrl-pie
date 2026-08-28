from __future__ import annotations

from collections.abc import Generator
from functools import lru_cache

from fastapi import HTTPException, status
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from ctrl_pi.config import get_config


class Base(DeclarativeBase):
    pass


def normalize_database_url(url: str) -> str:
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


@lru_cache(maxsize=4)
def engine_for_url(url: str) -> Engine:
    return create_engine(normalize_database_url(url), pool_pre_ping=True)


def configured_engine() -> Engine | None:
    value = get_config().database_url
    if value is None or not value.get_secret_value().strip():
        return None
    return engine_for_url(value.get_secret_value())


def get_db() -> Generator[Session, None, None]:
    engine = configured_engine()
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PostgreSQL is not configured. Set DATABASE_URL to continue.",
        )

    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        yield session

