import json
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from app import app

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
