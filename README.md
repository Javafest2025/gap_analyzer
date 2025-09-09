# Gap Analyzer

A FastAPI-based service for analyzing research gaps in academic papers and extracting structured content.

## Features

- **PDF Processing**: Extract structured content from academic papers
- **Database Storage**: Store extracted content in PostgreSQL
- **REST API**: Comprehensive API for managing extractions
- **Health Monitoring**: Built-in health check endpoints
- **Async Processing**: High-performance async/await architecture

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 13+
- Docker (optional, for external services)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd gap-analyzer
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp env.example .env
   # Edit .env with your database credentials
   ```

5. **Set up database**
   ```bash
   # Create PostgreSQL database
   createdb gap_analyzer
   ```

6. **Run the application**
   ```bash
   python -m app.main
   ```

### Using the Local Script

The project includes a convenient local development script:

```bash
# Complete setup
./scripts/local.sh setup

# Install dependencies
./scripts/local.sh install

# Run the server
./scripts/local.sh run

# Run tests
./scripts/local.sh test

# Format code
./scripts/local.sh format

# Run linting
./scripts/local.sh lint
```

## API Documentation

Once the server is running, you can access:

- **Swagger UI**: http://localhost:8003/api/v1/docs
- **ReDoc**: http://localhost:8003/api/v1/redoc
- **Health Check**: http://localhost:8003/api/v1/health

## Project Structure

```
gap-analyzer/
├── app/
│   ├── api/                 # API endpoints
│   │   ├── health.py       # Health check endpoints
│   │   └── extraction.py   # Extraction endpoints
│   ├── core/               # Core functionality
│   │   └── database.py     # Database configuration
│   ├── model/              # Database models
│   │   ├── base.py         # Base model class
│   │   ├── paper.py        # Paper model
│   │   └── paper_extraction.py  # Extraction models
│   ├── services/           # Business logic services
│   ├── utils/              # Utility functions
│   ├── config.py           # Configuration settings
│   └── main.py             # FastAPI application
├── scripts/
│   └── local.sh            # Local development script
├── requirements.txt        # Python dependencies
├── env.example            # Environment variables template
└── README.md              # This file
```

## Environment Variables

Key environment variables (see `env.example` for complete list):

- `DB_HOST`: PostgreSQL host (default: localhost)
- `DB_PORT`: PostgreSQL port (default: 5432)
- `DB_NAME`: Database name (default: gap_analyzer)
- `DB_USER`: Database user (default: postgres)
- `DB_PASSWORD`: Database password
- `API_HOST`: API host (default: 0.0.0.0)
- `API_PORT`: API port (default: 8003)
- `GEMINI_API_KEY`: Google Gemini API key (optional)

## Development

### Code Quality

The project uses several tools for code quality:

- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **Pytest**: Testing

Run all quality checks:
```bash
./scripts/local.sh format
./scripts/local.sh lint
./scripts/local.sh test
```

### Database Models

The application uses SQLAlchemy with async support. Models are based on the Spring backend schemas and include:

- `Paper`: Basic paper information
- `PaperExtraction`: Main extraction metadata
- `ExtractedSection`: Document sections with hierarchy
- `ExtractedFigure`: Figures and images
- `ExtractedTable`: Tables with structure
- `ExtractedEquation`: Mathematical equations
- `ExtractedCodeBlock`: Code blocks and algorithms
- `ExtractedReference`: Bibliographic references
- `ExtractedEntity`: Named entities

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

[Add your license information here]
