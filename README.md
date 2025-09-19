# Sentiment ML Service 🤖

A high-performance sentiment analysis service built with FastAPI and deployed on Modal with GPU acceleration. This service provides real-time sentiment analysis for single texts and batch processing capabilities.

## ✨ Features

- 3-class sentiment (positive, neutral, negative) via RoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest)
- Remote GPU execution via Modal by default; optional lightweight local stub for tests/dev
- Batch processing with aggregation stats and label normalization
- Thread-safe lazy pipeline initialization
- Optional Redis caching for summaries and fast-analysis results
- Comprehensive tests with coverage and markers for external/integration

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Modal account (free tier available)
- Virtual environment (recommended)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/theresaanna/sentiment_ml_service.git
cd sentiment_ml_service

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests locally
python -m pytest tests/ -v
```

### Run the API locally (development)

By default the service calls a remote GPU on Modal. For local development without model downloads, enable the lightweight fake pipeline and start FastAPI via Uvicorn:

```bash
export USE_FAKE_PIPELINE=1
uvicorn app:app --reload
```

To exercise the remote GPU path from local runs, leave USE_FAKE_PIPELINE unset (or set to anything other than 1) and configure Modal as below.

### Modal Setup

```bash
# Install Modal
pip install modal

# Authenticate with Modal
modal setup

# Deploy using the deployment script
python deploy.py

# Or deploy directly
python modal_app.py
```

## 📝 API Documentation

### Base URL
```
https://theresaanna--sentiment-ml-service-fastapi-app.modal.run
```

### Endpoints

#### Health Check
```http
GET /health
```

Response:
```json
{
  "status": "ok",
  "model": "cardiffnlp/twitter-roberta-base-sentiment-latest"
}
```

#### Analyze Single Text
```http
POST /analyze-text
Content-Type: application/json

{
  "text": "I absolutely love this product!",
  "method": "auto"  // optional
}
```

Response:
```json
{
  "success": true,
  "result": {
    "predicted_sentiment": "positive",
    "confidence": 0.9998
  },
  "models_used": ["roberta"]
}
```

#### Analyze Batch
```http
POST /analyze-batch
Content-Type: application/json

{
  "texts": [
    "This is amazing!",
    "I hate this",
    "It's okay I guess"
  ],
  "method": "auto"  // optional
}
```

Response:
```json
{
  "success": true,
  "results": [
    {
      "text": "This is amazing!",
      "predicted_sentiment": "positive",
      "confidence": 0.9997
    },
    {
      "text": "I hate this",
      "predicted_sentiment": "negative",
      "confidence": 0.9995
    },
    {
      "text": "It's okay I guess",
      "predicted_sentiment": "neutral",
      "confidence": 0.8234
    }
  ],
  "statistics": {
    "total_analyzed": 3,
    "sentiment_distribution": {
      "positive": 1,
      "negative": 1,
      "neutral": 1
    }
  },
  "overall_sentiment": "neutral"
}
```

## 🧪 Python Client (optional)

You can call the service from Python using the provided client.

Environment variables:
- MODAL_ML_BASE_URL: Required base URL for the service (e.g., https://your-app.modal.run)
- MODAL_ML_API_KEY: Optional API key sent as a Bearer token

Example:

```python path=null start=null
from ml_service_client import MLServiceClient

# Base URL can come from MODAL_ML_BASE_URL env var or be passed directly
client = MLServiceClient(base_url="https://theresaanna--sentiment-ml-service-fastapi-app.modal.run")

# Single text
res = client.analyze_text("I absolutely love this!")
print(res)

# Batch
texts = ["This is amazing!", "I hate this", "It's okay I guess"]
batch = client.analyze_batch(texts)
print(batch["statistics"], batch["results"][0])
```

## 🧪 Testing

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Run Specific Test Suite
```bash
# Basic tests
python -m pytest tests/test_app.py -v

# Comprehensive tests
python -m pytest tests/test_sentiment_analysis.py -v

# Performance tests only
python -m pytest tests/test_sentiment_analysis.py::TestPerformance -v
```

### Test markers and external dependencies
- Markers:
  - integration: tests that hit external services or require secrets
  - external: tests that call external APIs (e.g., Modal, OpenAI)
- To skip them (recommended for local dev):
```bash
pytest -m "not integration and not external" -vv
```
- OpenAI summarizer tests automatically skip if OPENAI_API_KEY is not set.
- Neutral detection tests skip unless a local service is reachable at LOCAL_API_URL (default http://localhost:8000).
- To test RoBERTa directly, set RUN_ROBERTA_TESTS=1 (may download a model). To use deployed GPU health check instead, set MODAL_GPU_TEST=1.

### Test Coverage
- Quick view (skips external/integration):
```bash
pytest -m "not integration and not external" --cov=. --cov-branch --cov-report=term-missing:skip-covered
```
- HTML report:
```bash
pytest -m "not integration and not external" --cov=. --cov-branch --cov-report=html
open htmlcov/index.html
```

## 🚢 Deployment

### Automated Deployment

Use the provided deployment script for a complete deployment with testing:

```bash
python deploy.py
```

This script will:
1. Check Modal authentication
2. Run local tests
3. Deploy to Modal with GPU support
4. Run tests on Modal infrastructure
5. Verify deployment

### Manual Deployment

```bash
# Deploy to Modal
modal deploy modal_app.py::app

# Check deployment status
modal app list

# View logs
modal app logs sentiment-ml-service

# Stop the app
modal app stop sentiment-ml-service
```

## ⚙️ Configuration

### Environment Variables

- MODEL_NAME: Hugging Face model ID (default: cardiffnlp/twitter-roberta-base-sentiment-latest)
- USE_FAKE_PIPELINE: Set to 1 to use a lightweight local stub (avoids model download and remote calls) for dev/tests
- MODAL_APP_NAME: Modal app name for remote GPU functions (default: sentiment-ml-service)
- OPENAI_API_KEY: Required for the /summarize endpoint (OpenAI-powered CommentSummarizer)
- OPENAI_SUMMARY_MODEL: OpenAI model for summaries (default: gpt-4o-mini)
- OPENAI_TIMEOUT_SECONDS: HTTP timeout for OpenAI requests (default: 30)
- REDIS_URL: Optional Redis URL (enables caching if reachable)
- REDIS_CACHE_TTL_HOURS: Default cache TTL in hours (default: 24)
- REDIS_ANALYSIS_TTL_HOURS: Analysis-specific TTL in hours (default: 6)
- MODAL_ML_BASE_URL: Base URL for the optional Python client (ml_service_client)
- MODAL_ML_API_KEY: API key for the optional Python client
- OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS: Thread caps (default: 1)
- TOKENIZERS_PARALLELISM: Set to "false" to avoid tokenizer parallel warnings

### Modal Configuration

- **GPU**: NVIDIA T4 (cost-effective for inference)
- **Min Containers**: 1 (always warm)
- **Max Containers**: 10 (auto-scaling)
- **Timeout**: 120 seconds
- **Concurrent Inputs**: 100 per container

## 📊 Performance

- **Latency**: ~50-100ms per request (warm container)
- **Throughput**: Up to 1000 requests/second with auto-scaling
- **Batch Size**: Optimized for 32 texts per batch
- **Cold Start**: ~5-10 seconds (model loading)

## 💰 Cost Optimization

- Uses efficient DistilBERT model (40% smaller than BERT)
- T4 GPUs for optimal cost/performance ratio
- Auto-scaling to handle load spikes
- Minimum 1 container to avoid cold starts
- Batch processing for efficiency

## 🛠️ Development

### Project Structure
```
sentiment_ml_service/
├── app.py                     # FastAPI application (routes, remote GPU integration, summarize)
├── modal_app.py               # Modal deployment configuration
├── deploy.py                  # Automated deployment script
├── cache.py                   # Optional Redis cache service
├── ml_service_client.py       # Simple HTTP client for the service
├── app_modules/
│   ├── __init__.py
│   └── science/
│       └── comment_summarizer.py   # OpenAI-powered summarizer
├── tests/
│   ├── test_app.py
│   ├── test_sentiment_analysis.py
│   ├── test_app_additional.py
│   ├── test_cache_unit.py
│   ├── test_comment_summarizer_unit.py
│   └── test_deploy_unit.py
├── test_neutral_detection.py      # Optional local/Modal validation (skipped by default)
├── test_openai_summarizer.py      # OpenAI summarizer tests (skips if no API key)
├── test_modal_integration.py      # Modal integration test
├── requirements.txt
├── README.md
└── README_OPENAI_SETUP.md
```

### Adding New Features

1. Update `app.py` with new endpoints
2. Add tests in `tests/`
3. Update `modal_app.py` if needed
4. Run `python deploy.py` to deploy

## 🐛 Troubleshooting

### Common Issues

1. **Modal Authentication Failed**
   ```bash
   modal setup
   ```

2. **GPU Not Available**
   - Check Modal subscription (free tier includes GPU)
   - Verify CUDA installation in logs

3. **Tests Failing**
   ```bash
   # Run with verbose output
   python -m pytest tests/ -vv --tb=short
   ```

4. **Deployment Issues**
   ```bash
   # Check Modal logs
   modal app logs sentiment-ml-service
   ```

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## 📧 Support

For issues or questions:
- Open an issue on GitHub
- Contact: theresa@example.com

## 🔄 Updates

- **v1.3.0** (2025-09): Documentation refreshed (RoBERTa default, local dev via USE_FAKE_PIPELINE, expanded tests and coverage guidance), added Python client docs
- **v1.2.0** (2024-01): Performance optimizations and comprehensive testing
- **v1.1.0** (2024-01): Added batch processing and auto-scaling
- **v1.0.0** (2024-01): Initial release with GPU support
