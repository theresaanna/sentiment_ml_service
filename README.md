# Sentiment ML Service 🤖

A high-performance sentiment analysis service built with FastAPI and deployed on Modal with GPU acceleration. This service provides real-time sentiment analysis for single texts and batch processing capabilities.

## ✨ Features

- **GPU-Accelerated Inference**: Utilizes NVIDIA T4 GPUs for fast model inference
- **Auto-scaling**: Automatically scales from 1 to 10 containers based on demand
- **Batch Processing**: Efficiently analyze multiple texts in a single request
- **Thread-Safe**: Singleton pattern with thread locks for safe concurrent access
- **Production-Ready**: Comprehensive test suite and automated deployment
- **Cost-Optimized**: Uses efficient DistilBERT model with smart resource allocation

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
  "model": "distilbert-base-uncased-finetuned-sst-2-english"
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
  "models_used": ["distilbert"]
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

### Test Coverage
```bash
python -m pytest tests/ --cov=app --cov-report=html
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

| Variable | Description | Default |
|----------|-------------|---------||
| `MODEL_NAME` | Hugging Face model ID | `distilbert-base-uncased-finetuned-sst-2-english` |
| `CUDA_VISIBLE_DEVICES` | GPU device selection | `0` |
| `OMP_NUM_THREADS` | OpenMP thread limit | `1` |
| `MKL_NUM_THREADS` | MKL thread limit | `1` |
| `OPENBLAS_NUM_THREADS` | OpenBLAS thread limit | `1` |
| `TOKENIZERS_PARALLELISM` | Tokenizer parallelism | `false` |

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
├── app.py                 # FastAPI application
├── modal_app.py          # Modal deployment configuration
├── deploy.py             # Automated deployment script
├── requirements.txt      # Python dependencies
├── tests/
│   ├── test_app.py      # Basic tests
│   └── test_sentiment_analysis.py  # Comprehensive tests
└── README.md            # This file
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

- **v1.0.0** (2024-01): Initial release with GPU support
- **v1.1.0** (2024-01): Added batch processing and auto-scaling
- **v1.2.0** (2024-01): Performance optimizations and comprehensive testing
