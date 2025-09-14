import json
from fastapi.testclient import TestClient
import sys
import os
# Get the directory of the current file (test_app.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the directory containing the 'sentiment_ml_service' package (two levels up from 'tests')
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
# Add the project root to sys.path
sys.path.insert(0, project_root)
from sentiment_ml_service.app import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "model" in data

def test_analyze_text():
    r = client.post("/analyze-text", json={"text": "I love this"})
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    assert "result" in data
    assert "predicted_sentiment" in data["result"]


def test_analyze_batch_empty():
    r = client.post("/analyze-batch", json={"texts": []})
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    assert data["statistics"]["total_analyzed"] == 0
