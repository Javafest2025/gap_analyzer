"""
Database connection and session management.
app/core/database.py
"""

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy import text
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from loguru import logger

from app.core.config import settings


# Create declarative base
Base = declarative_base()


class DatabaseManager:
    """Manages database connections and sessions."""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or settings.database_url
        self._engine: Optional[AsyncEngine] = None
        self._sessionmaker: Optional[async_sessionmaker] = None
    
    async def initialize(self):
        """Initialize database engine and session maker."""
        import asyncio
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Initializing database connection (attempt {attempt + 1}/{max_retries})")
                
                # Create engine with connection pooling
                self._engine = create_async_engine(
                    self.database_url,
                    echo=settings.DB_ECHO,
                    pool_size=settings.DB_POOL_SIZE,
                    max_overflow=settings.DB_MAX_OVERFLOW,
                    pool_timeout=settings.DB_POOL_TIMEOUT,
                    pool_pre_ping=True,  # Verify connections before using
                    poolclass=QueuePool,
                    connect_args={
                        "server_settings": {
                            "application_name": "gap_analyzer",
                        }
                    }
                )
                
                # Create session maker
                self._sessionmaker = async_sessionmaker(
                    self._engine,
                    class_=AsyncSession,
                    expire_on_commit=False,
                    autocommit=False,
                    autoflush=False
                )
                
                # Test connection
                async with self._engine.begin() as conn:
                    await conn.execute(text("SELECT 1"))
                
                logger.info("Database connection initialized successfully")
                return
                
            except Exception as e:
                logger.error(f"Failed to initialize database (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error("All database initialization attempts failed")
                    raise
    
    async def close(self):
        """Close database connections."""
        if self._engine:
            await self._engine.dispose()
            logger.info("Database connections closed")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get an async database session with retry logic for DNS resolution failures.
        
        Yields:
            AsyncSession: Database session
        """
        import asyncio
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                if not self._sessionmaker:
                    await self.initialize()
                
                async with self._sessionmaker() as session:
                    try:
                        yield session
                        await session.commit()
                        return
                    except Exception as e:
                        await session.rollback()
                        # Check if it's a DNS resolution error
                        if "getaddrinfo failed" in str(e) or "Name or service not known" in str(e):
                            logger.warning(f"DNS resolution error in database session (attempt {attempt + 1}/{max_retries}): {e}")
                            if attempt < max_retries - 1:
                                logger.info(f"Retrying database session in {retry_delay} seconds...")
                                await asyncio.sleep(retry_delay)
                                retry_delay *= 2
                                continue
                        raise
                    finally:
                        await session.close()
                        
            except Exception as e:
                if "getaddrinfo failed" in str(e) or "Name or service not known" in str(e):
                    logger.warning(f"DNS resolution error in database connection (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying database connection in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                raise
    
    async def get_db(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Dependency for FastAPI to get database session.
        
        Yields:
            AsyncSession: Database session
        """
        async with self.get_session() as session:
            yield session
    
    async def health_check(self) -> bool:
        """
        Check database health.
        
        Returns:
            bool: True if database is healthy
        """
        try:
            async with self.get_session() as session:
                result = await session.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# Create global database manager instance
db_manager = DatabaseManager()


# Dependency for FastAPI
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency to get database session.
    
    Yields:
        AsyncSession: Database session
    """
    async with db_manager.get_session() as session:
        yield session


# Note: Database table creation/management is handled by Spring Boot service
# This Python service only performs CRUD operations on existing tables


# Export
__all__ = [
    'Base',
    'DatabaseManager',
    'db_manager',
    'get_db'
]
