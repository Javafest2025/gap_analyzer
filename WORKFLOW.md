# Gap Analyzer Workflow

## Current System Architecture

The Gap Analyzer is a microservice that processes academic papers to identify research gaps. Here's how the complete workflow operates:

## 1. System Startup & Initialization

```
Application Startup
├── Configuration Validation
│   ├── Database settings
│   ├── GROBID URL
│   ├── RabbitMQ credentials
│   └── Gemini API key
├── Database Initialization
│   ├── Create tables
│   └── Test connection
└── External Services Initialization
    ├── GROBID Client (PDF processing)
    ├── RabbitMQ Service (message queue)
    └── Health checks for all services
```

## 2. Message Processing Workflow

### Input: RabbitMQ Message
```json
{
  "paper_id": "uuid",
  "paper_extraction_id": "uuid", 
  "correlation_id": "string",
  "request_id": "string",
  "config": {}
}
```

### Processing Steps:

#### Step 1: Message Reception
```
RabbitMQ Service
├── Listen on queue: "gap_analysis_requests"
├── Parse incoming message
├── Create GapAnalysisRequest object
└── Start processing
```

#### Step 2: Gap Analysis Orchestration
```
GapAnalysisService.analyze_paper()
├── Create GapAnalysis record in database
├── Fetch paper data and extracted content
├── Generate initial gaps using Gemini AI
├── Process gaps in batches (concurrent)
└── Update analysis summary
```

#### Step 3: Individual Gap Processing
```
For each gap:
├── Create ResearchGap record
├── Validate gap (search for related work)
├── Expand gap details (if valid)
└── Return processed gap
```

#### Step 4: Gap Validation Process
```
Gap Validation
├── Generate search query using Gemini
├── Search academic databases:
│   ├── Semantic Scholar
│   ├── CrossRef
│   └── arXiv
├── Analyze search results with Gemini
├── Determine if gap is still valid
└── Update validation status
```

#### Step 5: Gap Expansion (for valid gaps)
```
Gap Expansion
├── Generate detailed information:
│   ├── Potential impact
│   ├── Research hints
│   ├── Implementation suggestions
│   ├── Risks and challenges
│   ├── Required resources
│   └── Estimated difficulty
└── Create research topics
```

#### Step 6: Response Generation
```
Response
├── Compile all valid gaps
├── Create GapAnalysisResponse
├── Publish to RabbitMQ response exchange
└── Log completion
```

## 3. Data Flow

### Input Data Sources:
- **Paper Entity**: Title, abstract, authors, publication info
- **PaperExtraction**: Extracted content from PDF via GROBID
  - Sections, paragraphs, figures, tables
  - Equations, code blocks, references
  - Entities and metadata

### Processing Services:
- **GeminiService**: AI-powered gap identification and validation
- **WebSearchService**: Academic paper search across multiple APIs
- **GrobidClient**: PDF text extraction (if needed)

### Output Data:
- **GapAnalysis**: Overall analysis metadata
- **ResearchGap**: Individual gap details with validation
- **GapTopic**: Suggested research topics
- **GapValidationPaper**: Papers used for validation

## 4. API Endpoints

### Health & Monitoring:
- `GET /api/v1/health` - Basic health check
- `GET /api/v1/health/detailed` - Detailed service status
- `GET /api/v1/ready` - Readiness check
- `GET /api/v1/live` - Liveness check

### Gap Analysis Management:
- `GET /api/v1/gap-analyses` - List analyses (paginated)
- `GET /api/v1/gap-analyses/{id}` - Get analysis details
- `GET /api/v1/gaps/{id}` - Get gap details
- `POST /api/v1/gap-analyses/{id}/retry` - Retry failed analysis

### Statistics:
- `GET /api/v1/stats` - Service statistics

## 5. Logging & Monitoring

### Log Files:
- `logs/gap_analysis.log` - Main application logs
- `logs/grobid.log` - GROBID service logs
- `logs/rabbitmq.log` - RabbitMQ service logs
- `logs/gemini.log` - Gemini API logs

### Health Monitoring:
- Database connectivity
- GROBID service status
- RabbitMQ connection status
- Gemini API configuration

## 6. Error Handling & Resilience

### Retry Mechanisms:
- External API calls (3 retries with delays)
- Database operations
- Message processing

### Rate Limiting:
- Gemini API calls (15 requests/minute)
- Search API calls
- GROBID requests

### Batch Processing:
- Concurrent gap processing (5 gaps per batch)
- Maximum 2 concurrent batches
- Graceful error handling per gap

## 7. Configuration

### Required Environment Variables:
```
# Database
DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

# External Services  
GROBID_URL, GEMINI_API_KEY

# RabbitMQ
RABBITMQ_USER, RABBITMQ_PASSWORD, RABBITMQ_HOST, RABBITMQ_PORT

# Optional
DEBUG, LOG_LEVEL, API_HOST, API_PORT
```

## 8. Current Limitations

- No direct PDF upload endpoint (relies on pre-extracted content)
- No real-time progress updates
- No gap analysis result caching
- No user authentication/authorization
- No rate limiting on API endpoints

## 9. Future Enhancements

- Direct PDF processing endpoint
- WebSocket for real-time progress
- Redis caching for results
- User management and authentication
- Advanced gap categorization
- Citation network analysis
