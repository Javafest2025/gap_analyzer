# Gap Analysis Service - Setup and Usage Guide

## Project Structure

```
gap-analysis-service/
├── app/
│   ├── __init__.py
│   ├── main.py                      # Main application entry point
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py                # Configuration settings
│   ├── model/
│   │   ├── __init__.py
│   │   ├── base.py                  # Base model (existing)
│   │   ├── paper_models.py          # Paper model (existing)
│   │   ├── paper_extraction_models.py # Extraction models (existing)
│   │   └── gap_models.py            # Gap analysis models (new)
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── gap_schemas.py           # Pydantic schemas for gap analysis
│   ├── services/
│   │   ├── __init__.py
│   │   ├── gap_analysis_service.py  # Main gap analysis orchestrator
│   │   ├── gemini_service.py        # Gemini AI integration
│   │   ├── search_service.py        # Web search service
│   │   ├── grobid_client.py         # GROBID client
│   │   └── rabbitmq_service.py      # RabbitMQ consumer/publisher
│   └── utils/
│       └── __init__.py
├── alembic/
│   ├── versions/
│   │   └── add_gap_analysis_tables.py  # Migration for gap tables
│   └── alembic.ini
├── logs/                             # Log files directory
├── requirements.txt
├── .env
└── README.md
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Ensure your `.env` file has all required variables:

```env
# Database
DB_HOST=localhost
DB_PORT=5435
DB_NAME=projectDB
DB_USER=scholar
DB_PASSWORD=

# RabbitMQ
RABBITMQ_USER=scholar
RABBITMQ_PASSWORD=
RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672

# External Services
GROBID_URL=http://localhost:8070
GEMINI_API_KEY=
```

### 3. Run Database Migrations

```bash
# Initialize Alembic if not already done
alembic init alembic

# Generate migration
alembic revision --autogenerate -m "Add gap analysis tables"

# Apply migrations
alembic upgrade head
```

### 4. Start Required Services

Ensure these services are running:
- PostgreSQL (port 5435)
- RabbitMQ (port 5672)
- GROBID (port 8070)

### 5. Run the Gap Analysis Service

```bash
python app/main.py
```

## How It Works

### Message Flow

1. **Spring Backend → RabbitMQ**: Sends gap analysis request with paper details
2. **Gap Analysis Service**: Consumes message from `gap_analysis_requests` queue
3. **Processing Pipeline**:
   - Fetch paper and extraction data from database
   - Generate initial gaps using Gemini AI
   - Validate each gap by searching for related papers
   - Extract content from found papers using GROBID
   - Validate if gap is still valid using AI analysis
   - Expand valid gaps with detailed information
   - Generate research topic suggestions
4. **Response → RabbitMQ**: Publishes results to `gap_analysis_responses` exchange
5. **Spring Backend**: Receives and processes the response

### Request Format (from Spring)

```json
{
  "paper_id": "uuid-of-paper",
  "paper_extraction_id": "uuid-of-extraction",
  "correlation_id": "unique-correlation-id",
  "request_id": "unique-request-id",
  "config": {
    "max_gaps": 10,
    "validation_depth": "thorough"
  }
}
```

### Response Format (to Spring)

```json
{
  "request_id": "unique-request-id",
  "correlation_id": "unique-correlation-id",
  "status": "SUCCESS",
  "message": "Successfully identified 7 valid research gaps",
  "gap_analysis_id": "uuid-of-analysis",
  "total_gaps": 10,
  "valid_gaps": 7,
  "gaps": [
    {
      "gap_id": "uuid",
      "name": "Gap name",
      "description": "Detailed description",
      "category": "methodological",
      "validation_status": "VALID",
      "confidence_score": 0.85,
      "potential_impact": "High impact on...",
      "research_hints": "Consider exploring...",
      "implementation_suggestions": "Start by...",
      "risks_and_challenges": "Main challenges include...",
      "required_resources": "Requires expertise in...",
      "estimated_difficulty": "medium",
      "estimated_timeline": "6-12 months",
      "evidence_anchors": [
        {
          "title": "Paper Title",
          "url": "https://...",
          "type": "supporting"
        }
      ],
      "suggested_topics": [
        {
          "title": "Research Topic",
          "description": "Topic description",
          "research_questions": ["Q1", "Q2"],
          "methodology_suggestions": "Use approach X",
          "expected_outcomes": "Expected to achieve...",
          "relevance_score": 0.9
        }
      ]
    }
  ],
  "completed_at": "2025-01-10T12:00:00Z"
}
```

## Key Features

### 1. Intelligent Gap Generation
- Analyzes paper content, abstract, figures, and tables
- Identifies limitations, future work, and unexplored areas
- Categorizes gaps (theoretical, methodological, empirical, etc.)

### 2. Robust Gap Validation
- Searches multiple academic databases (Semantic Scholar, CrossRef, arXiv)
- Extracts and analyzes related papers
- Determines if gaps have been addressed
- Modifies gaps based on new findings

### 3. Comprehensive Gap Details
- Potential impact assessment
- Implementation suggestions
- Risk analysis
- Resource requirements
- Timeline estimates
- Research topic suggestions

### 4. Evidence-Based Analysis
- Links to supporting/conflicting papers
- Confidence scores for each gap
- Validation reasoning
- Paper analysis count

## Monitoring and Debugging

### Logs
- Application logs are stored in `logs/gap_analysis_YYYY-MM-DD.log`
- Console output shows real-time processing status

### Database Tables
- `gap_analyses`: Main analysis records
- `research_gaps`: Individual gaps with details
- `gap_validation_papers`: Papers analyzed for validation
- `gap_topics`: Suggested research topics

### RabbitMQ Queues
- Request Queue: `gap_analysis_requests`
- Response Exchange: `gap_analysis_responses`
- Routing Key: `gap.analysis.response`

## Performance Considerations

### Optimization Tips
1. **Batch Processing**: The service processes papers analyzed during validation in batches
2. **Async Operations**: All I/O operations are asynchronous
3. **Caching**: Consider adding Redis for caching search results
4. **Rate Limiting**: Be mindful of API rate limits (especially Gemini)

### Scalability
- Can run multiple instances for parallel processing
- RabbitMQ ensures message distribution
- Database connections are pooled

## Troubleshooting

### Common Issues

1. **GROBID Connection Error**
   - Ensure GROBID is running on the configured port
   - Check `GROBID_URL` in `.env`

2. **Gemini API Errors**
   - Verify API key is valid
   - Check quota limits
   - Monitor rate limits

3. **RabbitMQ Connection Failed**
   - Verify RabbitMQ is running
   - Check credentials in `.env`
   - Ensure queues/exchanges exist

4. **Database Connection Issues**
   - Verify PostgreSQL is running
   - Check database credentials
   - Ensure migrations are applied

## API Rate Limits

- **Gemini**: Free tier allows 15 requests per minute
- **Semantic Scholar**: 100 requests per 5 minutes
- **CrossRef**: No hard limit but be respectful
- **arXiv**: Max 1 request per 3 seconds

## Future Enhancements

1. **Caching Layer**: Add Redis for caching search results and extracted content
2. **Web UI**: Build a dashboard for monitoring gap analysis progress
3. **ML Models**: Train custom models for gap identification
4. **Citation Network**: Analyze citation networks for deeper gap validation
5. **Collaboration Features**: Allow multiple researchers to review gaps
6. **Export Formats**: Generate reports in various formats (PDF, LaTeX, Word)