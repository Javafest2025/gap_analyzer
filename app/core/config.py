"""
Core configuration module for the Gap Analysis Service.
app/core/config.py
"""

from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with validation."""
    
    # Application (hardcoded)
    APP_NAME: str = "Gap Analysis Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API Settings (hardcoded)
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8003
    API_PREFIX: str = "/api/v1"
    
    # Database Configuration
    DB_HOST: str
    DB_PORT: int = 5432
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str
    # Hardcoded database pool settings
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30
    DB_ECHO: bool = False
    
    # RabbitMQ Configuration
    RABBITMQ_USER: str
    RABBITMQ_PASSWORD: str
    RABBITMQ_HOST: str = "localhost"
    RABBITMQ_PORT: int = 5672
    # Hardcoded RabbitMQ settings
    RABBITMQ_VHOST: str = "/"
    
    # Queue Configuration (hardcoded - not used from settings)
    GAP_REQUEST_QUEUE: str = "gap_analysis_requests"
    GAP_RESPONSE_EXCHANGE: str = "gap_analysis_responses"
    GAP_RESPONSE_ROUTING_KEY: str = "gap.analysis.response"
    
    # External Services
    GROBID_URL: str
    # Hardcoded GROBID timeout
    GROBID_TIMEOUT: int = 120  # Increased timeout for GROBID processing
    GA_GEMINI_API_KEY: str
    GA_GEMINI_MODEL: str = "gemini-2.0-flash-exp"
    GA_GEMINI_RATE_LIMIT: int = 2  # requests per minute (extremely conservative for free tier)
    
    # Search Service Configuration (hardcoded - URLs not used in code)
    # SEMANTIC_SCHOLAR_API_URL: str = "https://api.semanticscholar.org/graph/v1/paper/search"
    # CROSSREF_API_URL: str = "https://api.crossref.org/works"
    # ARXIV_API_URL: str = "https://export.arxiv.org/api/query"
    # Hardcoded search settings
    SEARCH_MAX_RESULTS: int = 10
    SEARCH_TIMEOUT: int = 30
    
    # Gap Analysis Configuration (hardcoded)
    MAX_GAPS_PER_PAPER: int = 7
    MIN_GAPS_PER_PAPER: int = 3
    GAP_VALIDATION_PAPERS: int = 5
    GAP_CONFIDENCE_THRESHOLD: float = 0.5
    VALIDATION_BATCH_SIZE: int = 5
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/gap_analysis.log"
    # Hardcoded log settings
    LOG_ROTATION: str = "100 MB"
    LOG_RETENTION: str = "10 days"
    LOG_FORMAT: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    
    # Performance Configuration (hardcoded)
    ASYNC_TIMEOUT: int = 300  # 5 minutes
    MAX_CONCURRENT_VALIDATIONS: int = 1  # Process gaps sequentially to avoid rate limits
    RETRY_ATTEMPTS: int = 3
    RETRY_DELAY: int = 5  # seconds
    
    # Cache Configuration (hardcoded for future Redis integration)
    CACHE_ENABLED: bool = False
    CACHE_HOST: str = "localhost"
    CACHE_PORT: int = 6379
    CACHE_DB: int = 0
    CACHE_TTL: int = 3600  # 1 hour
    
    @property
    def database_url(self) -> str:
        """Construct database URL."""
        import urllib.parse
        # URL encode the password to handle special characters like @
        encoded_password = urllib.parse.quote(self.DB_PASSWORD, safe='')
        return f"postgresql+asyncpg://{self.DB_USER}:{encoded_password}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    @property
    def sync_database_url(self) -> str:
        """Construct synchronous database URL for Alembic."""
        import urllib.parse
        # URL encode the password to handle special characters like @
        encoded_password = urllib.parse.quote(self.DB_PASSWORD, safe='')
        return f"postgresql://{self.DB_USER}:{encoded_password}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    @property
    def rabbitmq_url(self) -> str:
        """Construct RabbitMQ URL with proper URL encoding."""
        import urllib.parse
        encoded_password = urllib.parse.quote(self.RABBITMQ_PASSWORD or "", safe="")
        encoded_vhost = urllib.parse.quote(self.RABBITMQ_VHOST or "/", safe="")
        return f"amqp://{self.RABBITMQ_USER}:{encoded_password}@{self.RABBITMQ_HOST}:{self.RABBITMQ_PORT}/{encoded_vhost}"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        
        # Allow extra fields for flexibility
        extra = "allow"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Settings: Application settings
    """
    return Settings()


# Create a global settings instance
settings = get_settings()


# Validation functions
def validate_settings():
    """Validate critical settings on startup."""
    errors = []
    
    # Check required API keys
    if not settings.GA_GEMINI_API_KEY:
        errors.append("GA_GEMINI_API_KEY is required")
    
    # Check database connection
    if not all([settings.DB_HOST, settings.DB_USER, settings.DB_PASSWORD, settings.DB_NAME]):
        errors.append("Database configuration is incomplete")
    
    # Check RabbitMQ configuration
    if not all([settings.RABBITMQ_USER, settings.RABBITMQ_PASSWORD]):
        errors.append("RabbitMQ configuration is incomplete")
    
    # Check GROBID URL
    if not settings.GROBID_URL:
        errors.append("GROBID_URL is required")
    
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
    return True


# Export commonly used configurations
__all__ = [
    'settings',
    'get_settings',
    'validate_settings',
    'Settings'
]
