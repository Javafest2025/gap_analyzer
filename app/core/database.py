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
        try:
            # Create engine with connection pooling
            self._engine = create_async_engine(
                self.database_url,
                echo=settings.DB_ECHO,
                pool_size=settings.DB_POOL_SIZE,
                max_overflow=settings.DB_MAX_OVERFLOW,
                pool_timeout=settings.DB_POOL_TIMEOUT,
                pool_pre_ping=True,  # Verify connections before using
                poolclass=QueuePool
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
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def close(self):
        """Close database connections."""
        if self._engine:
            await self._engine.dispose()
            logger.info("Database connections closed")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get an async database session.
        
        Yields:
            AsyncSession: Database session
        """
        if not self._sessionmaker:
            await self.initialize()
        
        async with self._sessionmaker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
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


# Utility functions
async def init_db():
    """Initialize database tables."""
    try:
        # First initialize the database manager if not already done
        if not db_manager._engine:
            await db_manager.initialize()
        
        async with db_manager._engine.begin() as conn:
            # Import all models to ensure they're registered
            from app.model import (
                paper,
                paper_extraction,
                gap_models
            )
            
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables initialized")
            
    except Exception as e:
        logger.error(f"Failed to initialize database tables: {e}")
        raise


async def drop_db():
    """Drop all database tables (use with caution!)."""
    try:
        # First initialize the database manager if not already done
        if not db_manager._engine:
            await db_manager.initialize()
            
        async with db_manager._engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            logger.warning("All database tables dropped!")
            
    except Exception as e:
        logger.error(f"Failed to drop database tables: {e}")
        raise


# Export
__all__ = [
    'Base',
    'DatabaseManager',
    'db_manager',
    'get_db',
    'init_db',
    'drop_db'
]
