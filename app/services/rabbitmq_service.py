"""
RabbitMQ service for consuming and publishing messages.
"""

import json
import asyncio
from typing import Optional, Dict, Any
from aio_pika import connect_robust, Message, ExchangeType, DeliveryMode
from aio_pika.abc import AbstractIncomingMessage, AbstractConnection, AbstractChannel
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.gap_schemas import GapAnalysisRequest, GapAnalysisResponse
from app.services.gap_analysis_service import GapAnalysisService
from app.core.config import settings
from app.core.database import db_manager


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
        
        # Use the improved database manager with retry logic
        # No need to create separate engine - use the global db_manager
        
        # Queue configuration
        self.request_queue = "gap_analysis_requests"
        self.request_exchange = "scholarai.exchange"
        self.request_routing_key = "gap.analysis.request"
        self.response_exchange = "gap_analysis_responses"
        self.response_routing_key = "gap.analysis.response"
    
    async def connect(self, retries: int = 10, delay: float = 1.0):
        """Connect to RabbitMQ and setup queues/exchanges with retry logic."""
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                # Use connect_robust with separate parameters to avoid URL encoding issues
                from app.core.config import settings
                self.connection = await connect_robust(
                    host=settings.RABBITMQ_HOST,
                    port=settings.RABBITMQ_PORT,
                    login=settings.RABBITMQ_USER,
                    password=settings.RABBITMQ_PASSWORD,
                    virtualhost=settings.RABBITMQ_VHOST,  # "/" is fine here
                )
                logger.info(f"Successfully connected to RabbitMQ on attempt {attempt}")
                break
            except Exception as e:
                last_err = e
                logger.warning(f"RabbitMQ connect failed ({attempt}/{retries}): {e}; retrying in {delay}s")
                await asyncio.sleep(delay)
                delay = min(delay * 2, 15)  # Exponential backoff, max 15s
        else:
            logger.error(f"Exhausted retries connecting to RabbitMQ: {last_err}")
            raise last_err
        
        try:
            self.channel = await self.connection.channel()
            
            # Set prefetch count to process one message at a time
            await self.channel.set_qos(prefetch_count=1)
            
            # Declare request exchange (the main application exchange)
            request_exchange = await self.channel.declare_exchange(
                self.request_exchange,
                type=ExchangeType.TOPIC,
                durable=True
            )
            
            # Declare request queue
            request_queue = await self.channel.declare_queue(
                self.request_queue,
                durable=True
            )
            
            # Bind request queue to the exchange with routing key
            await request_queue.bind(
                request_exchange,
                routing_key=self.request_routing_key
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
        """Process incoming gap analysis request with proper exception handling."""
        # Use requeue=False to reject messages on exception instead of requeuing
        async with message.process(requeue=False):
            try:
                # Parse message
                body = message.body.decode()
                logger.info(f"Received message: {body[:200]}...")
                
                request_data = json.loads(body)
                request = GapAnalysisRequest(**request_data)
                
                logger.info(f"Processing gap analysis for paper: {request.paperId}")
                
                # Create database session using the improved database manager with retry logic
                async with db_manager.get_session() as session:
                    try:
                        # Perform gap analysis
                        response = await self.gap_service.analyze_paper(request, session)
                        
                        # Publish response
                        await self.publish_response(response)
                        
                        logger.info(f"Gap analysis completed for request: {request.requestId}")
                        # Message will be acked automatically on successful exit
                        
                    except Exception as db_error:
                        # Rollback the session on any database error
                        await session.rollback()
                        logger.error(f"Database error during gap analysis: {db_error}")
                        
                        # Check if it's a duplicate correlation_id (idempotency issue)
                        if "duplicate key value violates unique constraint" in str(db_error) and "correlation_id" in str(db_error):
                            logger.info(f"Duplicate correlation_id {request.correlationId} - treating as retry")
                            
                            # Load existing analysis and return success
                            from sqlalchemy import select
                            from app.model.gap_models import GapAnalysis
                            
                            existing_analysis = await session.scalar(
                                select(GapAnalysis).where(GapAnalysis.correlation_id == request.correlationId)
                            )
                            
                            if existing_analysis:
                                # Create a success response for the existing analysis
                                response = GapAnalysisResponse(
                                    request_id=request.requestId,
                                    correlation_id=request.correlationId,
                                    status="SUCCESS",
                                    message="Analysis already exists (duplicate request handled)",
                                    gap_analysis_id=str(existing_analysis.id),
                                    total_gaps=existing_analysis.total_gaps_identified or 0,
                                    valid_gaps=existing_analysis.valid_gaps_count or 0,
                                    gaps=[]  # Could be populated if needed
                                )
                                
                                await self.publish_response(response)
                                # Message will be acked automatically on successful exit
                                return
                        
                        # For other database errors, re-raise to let context manager handle rejection
                        raise
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in message: {e}")
                await self._publish_error_response(
                    message.body.decode(),
                    f"Invalid JSON: {str(e)}"
                )
                # Don't re-raise - let context manager ack the message
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                
                # Try to extract request ID for error response
                try:
                    partial_data = json.loads(message.body.decode())
                    request_id = partial_data.get('requestId', 'unknown')
                    correlation_id = partial_data.get('correlationId', 'unknown')
                    
                    error_response = GapAnalysisResponse(
                        request_id=request_id,
                        correlation_id=correlation_id,
                        status="FAILED",
                        message=f"Processing failed: {str(e)}",
                        error=str(e)
                    )
                    
                    await self.publish_response(error_response)
                    # Don't re-raise - let context manager ack the message
                    
                except Exception as e:
                    logger.error(f"Could not send error response: {e}")
                    # Re-raise to let context manager reject the message
                    raise
    
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
                correlation_id=response.correlationId,
                headers={
                    'request_id': response.requestId,
                    'status': response.status
                }
            )
            
            # Publish message
            await exchange.publish(
                message,
                routing_key=self.response_routing_key
            )
            
            logger.info(f"Published response for request: {response.requestId}")
            
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
        
        # Close database connections using the global db_manager
        await db_manager.close()
        logger.info("Database connections closed")


def create_rabbitmq_service(settings) -> RabbitMQService:
    """Factory function to create RabbitMQ service using Settings.rabbitmq_url."""
    return RabbitMQService(
        rabbitmq_url=settings.rabbitmq_url,
        db_url="",  # Not used anymore - we use the global db_manager
        gemini_api_key=settings.GA_GEMINI_API_KEY,
        grobid_url=settings.GROBID_URL
    )