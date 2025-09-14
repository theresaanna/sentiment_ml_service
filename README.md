# Sentiment ML Service (Modal)

This repository hosts a lightweight, low-cost sentiment inference service designed to run on Modal. It exposes simple HTTP endpoints for text and batch sentiment analysis and is optimized for CPU-first usage and cold-start minimization.

Endpoints
- GET /health: Basic status with model info
- POST /analyze-text: Analyze a single text
- POST /analyze-batch: Analyze a list of texts

Request/Response examples
- POST /analyze-text
  Body: {"text": "I love this!", "method": "auto"}
  Response: {"success": true, "result": {"predicted_sentiment": "positive", "confidence": 0.98}, "models_used": ["distilbert"]}

- POST /analyze-batch
  Body: {"texts": ["good", "bad"], "method": "auto"}
  Response: {"success": true, "results": [...], "statistics": {"total_analyzed": 2, ...}, "overall_sentiment": "neutral"}

Local development
1) Install deps
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt

2) Run unit tests (heavy tests skipped by default)
   pytest -q

Modal deployment
Prereqs: Install and authenticate Modal CLI
- pip install modal
- modal token new

Deploy
- modal deploy modal_app.py

After deploy, Modal will output a public base URL for the FastAPI app, e.g.:
  https://sentiment-ml-service--<your-handle>.modal.run

Configure your Flask app with this URL
- Set MODAL_ML_BASE_URL to the URL above
- Optionally set MODAL_ML_API_KEY if you add auth later

Environment variables (optional)
- MODEL_NAME: HF model id (default: distilbert-base-uncased-finetuned-sst-2-english)
- HF_HOME / HF_HUB_CACHE: Cache directory config
- OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS: Thread caps

Notes
- This service focuses on inference. Training/feedback endpoints can be added later.
- For lowest cost: keep GPU off, use small distilled models, and allow scale-to-zero.
