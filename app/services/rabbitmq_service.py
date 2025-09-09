"""
RabbitMQ service for consuming and publishing messages.
"""

import json
import asyncio
from typing import Optional, Dict, Any
from aio_pika import connect, Message, ExchangeType, DeliveryMode
from aio_pika.abc import AbstractIncomingMessage, AbstractConnection, AbstractChannel
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.schemas.gap_schemas import GapAnalysisRequest, GapAnalysisResponse
from app.services.gap_analysis_service import GapAnalysisService
from app.core.config import settings


class RabbitMQService:
    """Service for handling RabbitMQ communications."""
    
    def __init__(
        self,
        rabbitmq_url: str,
        db_url: str,
        gemini_api_key: str,
        grobid_url: str
    ):
        self.rabbitmq_url = rabbitmq_url
        self.connection: Optional[AbstractConnection] = None
        self.channel: Optional[AbstractChannel] = None
        
        # Initialize services
        self.gap_service = GapAnalysisService(gemini_api_key, grobid_url)
        
        # Database setup
        self.engine = create_async_engine(db_url, echo=False)
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Queue configuration
        self.request_queue = "gap_analysis_requests"
        self.response_exchange = "gap_analysis_responses"
        self.response_routing_key = "gap.analysis.response"
    
    async def connect(self):
        """Connect to RabbitMQ and setup queues/exchanges."""
        try:
            # Connect to RabbitMQ
            self.connection = await connect(self.rabbitmq_url)
            self.channel = await self.connection.channel()
            
            # Set prefetch count to process one message at a time
            await self.channel.set_qos(prefetch_count=1)
            
            # Declare request queue
            request_queue = await self.channel.declare_queue(
                self.request_queue,
                durable=True
            )
            
            # Declare response exchange
            await self.channel.declare_exchange(
                self.response_exchange,
                type=ExchangeType.TOPIC,
                durable=True
            )
            
            # Set up consumer
            await request_queue.consume(self.process_message)
            
            logger.info(f"Connected to RabbitMQ and listening on queue: {self.request_queue}")
            
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise
    
    async def process_message(self, message: AbstractIncomingMessage):
        """Process incoming gap analysis request."""
        async with message.process():
            try:
                # Parse message
                body = message.body.decode()
                logger.info(f"Received message: {body[:200]}...")
                
                request_data = json.loads(body)
                request = GapAnalysisRequest(**request_data)
                
                logger.info(f"Processing gap analysis for paper: {request.paper_id}")
                
                # Create database session
                async with self.async_session() as session:
                    # Perform gap analysis
                    response = await self.gap_service.analyze_paper(request, session)
                
                # Publish response
                await self.publish_response(response)
                
                logger.info(f"Gap analysis completed for request: {request.request_id}")
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in message: {e}")
                await self._publish_error_response(
                    message.body.decode(),
                    f"Invalid JSON: {str(e)}"
                )
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                
                # Try to extract request ID for error response
                try:
                    partial_data = json.loads(message.body.decode())
                    request_id = partial_data.get('request_id', 'unknown')
                    correlation_id = partial_data.get('correlation_id', 'unknown')
                    
                    error_response = GapAnalysisResponse(
                        request_id=request_id,
                        correlation_id=correlation_id,
                        status="FAILED",
                        message=f"Processing failed: {str(e)}",
                        error=str(e)
                    )
                    
                    await self.publish_response(error_response)
                    
                except Exception as e:
                    logger.error(f"Could not send error response: {e}")
    
    async def publish_response(self, response: GapAnalysisResponse):
        """Publish gap analysis response to Spring backend."""
        try:
            if not self.channel:
                logger.error("Channel not initialized")
                return
            
            # Get exchange
            exchange = await self.channel.get_exchange(self.response_exchange)
            
            # Prepare message
            message_body = response.model_dump_json()
            message = Message(
                body=message_body.encode(),
                delivery_mode=DeliveryMode.PERSISTENT,
                content_type='application/json',
                correlation_id=response.correlation_id,
                headers={
                    'request_id': response.request_id,
                    'status': response.status
                }
            )
            
            # Publish message
            await exchange.publish(
                message,
                routing_key=self.response_routing_key
            )
            
            logger.info(f"Published response for request: {response.request_id}")
            
        except Exception as e:
            logger.error(f"Failed to publish response: {e}")
    
    async def _publish_error_response(self, original_message: str, error: str):
        """Publish error response when request parsing fails."""
        try:
            error_response = {
                'status': 'FAILED',
                'message': 'Failed to process request',
                'error': error,
                'original_message': original_message[:500]  # Truncate for safety
            }
            
            if self.channel:
                exchange = await self.channel.get_exchange(self.response_exchange)
                
                message = Message(
                    body=json.dumps(error_response).encode(),
                    delivery_mode=DeliveryMode.PERSISTENT,
                    content_type='application/json'
                )
                
                await exchange.publish(
                    message,
                    routing_key=self.response_routing_key
                )
                
        except Exception as e:
            logger.error(f"Failed to publish error response: {e}")
    
    async def start(self):
        """Start the RabbitMQ consumer."""
        await self.connect()
        
        try:
            # Keep the service running
            logger.info("Gap Analysis Service is running. Press Ctrl+C to stop.")
            await asyncio.Future()  # Run forever
            
        except KeyboardInterrupt:
            logger.info("Shutting down Gap Analysis Service...")
            
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the RabbitMQ consumer and cleanup."""
        if self.connection:
            await self.connection.close()
            logger.info("Disconnected from RabbitMQ")
        
        # Close database engine
        await self.engine.dispose()
        logger.info("Database connections closed")


async def create_rabbitmq_service(settings) -> RabbitMQService:
    """Factory function to create RabbitMQ service."""
    rabbitmq_url = f"amqp://{settings.RABBITMQ_USER}:{settings.RABBITMQ_PASSWORD}@localhost/"
    db_url = f"postgresql+asyncpg://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
    
    return RabbitMQService(
        rabbitmq_url=rabbitmq_url,
        db_url=db_url,
        gemini_api_key=settings.GEMINI_API_KEY,
        grobid_url=settings.GROBID_URL
    )