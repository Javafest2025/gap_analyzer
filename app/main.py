"""
Gap Analyzer - Main FastAPI application.
Provides API endpoints for analyzing research gaps in academic papers.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger
import sys

from app.core.config import settings
from app.api import health, endpoints
from app.core.database import db_manager

# Constants
APP_NAME = settings.APP_NAME
APP_VERSION = settings.APP_VERSION

# Global service instances
rabbitmq_service = None
grobid_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info(f"Starting {APP_NAME}...")
    
    # Validate settings first
    try:
        from app.core.config import validate_settings
        validate_settings()
        logger.info("Configuration validated successfully")
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        if not settings.DEBUG:
            sys.exit(1)
    
    # Initialize database connection (no table creation)
    try:
        await db_manager.initialize()
        logger.info("Database connection established successfully")
    except Exception as e:
        logger.error(f"Failed to establish database connection: {e}")
        if settings.DEBUG:
            logger.warning("Running in debug mode - continuing without database connection")
        else:
            logger.error("Exiting due to database connection failure")
            sys.exit(1)
    
    # Initialize and test external services
    try:
        await _initialize_external_services()
        logger.info("External services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize external services: {e}")
        if settings.DEBUG:
            logger.warning("Running in debug mode - continuing without external services")
        else:
            logger.error("Exiting due to external service initialization failure")
            sys.exit(1)
    
    yield
    
    # Shutdown
    logger.info(f"Shutting down {APP_NAME}...")
    await _cleanup_services()


# Create FastAPI application
app = FastAPI(
    title=APP_NAME,
    description="API for analyzing research gaps in academic papers and extracting structured content",
    version=APP_VERSION,
    docs_url="/docs",  # Root level docs
    redoc_url="/redoc",  # Root level redoc
    openapi_url="/openapi.json",  # Root level openapi
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logger.remove()

# Console logging with colors
logger.add(
    sys.stdout,
    level=settings.LOG_LEVEL,
    format=settings.LOG_FORMAT,
    colorize=True
)

# File logging
logger.add(
    settings.LOG_FILE,
    level=settings.LOG_LEVEL,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    rotation=settings.LOG_ROTATION,
    retention=settings.LOG_RETENTION,
    enqueue=True  # Thread-safe logging
)

# Add specific service loggers
logger.add(
    "logs/grobid.log",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | GROBID | {message}",
    rotation="50 MB",
    retention="7 days",
    filter=lambda record: "grobid" in record["name"].lower() or "grobid" in record["message"].lower()
)

logger.add(
    "logs/rabbitmq.log",
    level="DEBUG", 
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | RABBITMQ | {message}",
    rotation="50 MB",
    retention="7 days",
    filter=lambda record: "rabbitmq" in record["name"].lower() or "rabbitmq" in record["message"].lower()
)

logger.add(
    "logs/gemini.log",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | GEMINI | {message}",
    rotation="50 MB", 
    retention="7 days",
    filter=lambda record: "gemini" in record["name"].lower() or "gemini" in record["message"].lower()
)

# Include API routers
app.include_router(health.router, prefix=settings.API_PREFIX, tags=["Health"])
app.include_router(endpoints.router, prefix=settings.API_PREFIX, tags=["Endpoints"])


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": APP_NAME,
        "version": APP_VERSION,
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/api/v1/",
        "api_prefix": settings.API_PREFIX
    }


@app.get("/health")
async def root_health():
    """Root level health check for convenience."""
    return {
        "status": "healthy",
        "service": APP_NAME,
        "version": APP_VERSION
    }


async def _initialize_external_services():
    """Initialize and test external services."""
    global rabbitmq_service, grobid_client
    
    # Initialize GROBID client
    try:
        from app.services.grobid_client import GrobidClient
        grobid_client = GrobidClient(settings.GROBID_URL)
        logger.info(f"GROBID client initialized with URL: {settings.GROBID_URL}")
        
        # Test GROBID connection
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{settings.GROBID_URL}/api/isalive", timeout=5)
            if response.status_code == 200:
                logger.info("GROBID service is healthy and accessible")
            else:
                logger.warning(f"GROBID service returned status {response.status_code}")
                
    except Exception as e:
        logger.error(f"Failed to initialize GROBID client: {e}")
        raise
    
    # Initialize RabbitMQ service
    try:
        from app.services.rabbitmq_service import create_rabbitmq_service
        rabbitmq_service = create_rabbitmq_service(settings)
        logger.info(f"RabbitMQ service initialized with URL: {rabbitmq_service.rabbitmq_url}")
        logger.info(f"RabbitMQ connection params: host={settings.RABBITMQ_HOST}, port={settings.RABBITMQ_PORT}, user={settings.RABBITMQ_USER}, vhost={settings.RABBITMQ_VHOST}")
        
        # Test RabbitMQ connection with retry logic
        await rabbitmq_service.connect(retries=10, delay=2.0)
        logger.info("RabbitMQ service is healthy and accessible")
        
    except Exception as e:
        logger.error(f"Failed to initialize RabbitMQ service: {e}")
        raise


async def _cleanup_services():
    """Cleanup external services on shutdown."""
    global rabbitmq_service, grobid_client
    
    if rabbitmq_service:
        try:
            await rabbitmq_service.stop()
            logger.info("RabbitMQ service stopped")
        except Exception as e:
            logger.error(f"Error stopping RabbitMQ service: {e}")
    
    if grobid_client:
        try:
            await grobid_client.close()
            logger.info("GROBID client closed")
        except Exception as e:
            logger.error(f"Error closing GROBID client: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
