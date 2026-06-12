import os,  asyncio, logging, time
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from config import settings

os.makedirs("data", exist_ok=True)

# TESTING: Uses DATABASE_URL from config.py (defaults to localhost)
# PRODUCTION: Update DATABASE_URL in .env to point to managed PostgreSQL service

DATABASE_URL = settings.DATABASE_URL

# Connection pooling configuration (PRODUCTION: tune pool_size based on expected concurrency)
# TESTING defaults: pool_size=20 (suitable for dev)
# PRODUCTION: Increase pool_size = expected concurrent requests (e.g., 50-200 for production)
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_size=20,              # TESTING: 20 connections; PRODUCTION: 50-200+
    max_overflow=10,           # TESTING: 10 overflow; PRODUCTION: 20-50+
    pool_pre_ping=True,        # Verify connections before use (prevents "connection lost" errors)
    pool_recycle=3600,         # Recycle connections after 1 hour (prevents idle timeout)
    future=True,               # Use future mode for better async support
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    future=True,               # Use future mode for better async support
)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


async def init_db():
    from models import detection_event, face_event, person, watchlist, camera, journey  # noqa: F401
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
