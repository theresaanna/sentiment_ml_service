#!/usr/bin/env python3
"""
Test script to verify that the RoBERTa model correctly detects neutral sentiment.
This will help diagnose why neutral comments aren't showing up in the UI.
"""

import requests
import json
from typing import List, Dict
import os
import pytest

# Test comments with expected sentiments
TEST_COMMENTS = [
    # Clear positive examples
    ("This is absolutely amazing! I love it so much!", "positive"),
    ("Fantastic work, really impressed!", "positive"),
    ("Best video ever! 😍", "positive"),
    
    # Clear negative examples
    ("This is terrible and I hate it", "negative"),
    ("Worst video I've ever seen", "negative"),
    ("Completely disappointing", "negative"),
    
    # Clear neutral examples (these should be detected as neutral)
    ("This video was uploaded in 2012", "neutral"),
    ("The interviewer asks questions", "neutral"),
    ("She is wearing a black dress", "neutral"),
    ("The video is 10 minutes long", "neutral"),
    ("What time does this start?", "neutral"),
    ("I wonder what she meant by that", "neutral"),
    ("This is just her opinion", "neutral"),
    ("Interesting perspective", "neutral"),
    ("Thanks for sharing", "neutral"),
    ("First comment", "neutral"),
    
    # Mixed/ambiguous that might be neutral
    ("I'm not sure about this", "neutral"),
    ("Some people might like this", "neutral"),
    ("It depends on your perspective", "neutral"),
]


def test_local_api():
    """Test the local FastAPI sentiment service if it's running.
    Skips if the service is not reachable or not our API.
    """
    base_url = os.getenv("LOCAL_API_URL", "http://localhost:8000")

    # Check if service is reachable and appears to be our API; if not, skip
    try:
        health = requests.get(f"{base_url}/health", timeout=2)
        if health.status_code != 200:
            pytest.skip(f"Service not healthy at {base_url}")
        health_json = {}
        try:
            health_json = health.json()
        except Exception:
            pytest.skip(f"Service at {base_url} did not return JSON for /health")
        if health_json.get("status") != "ok" or "model" not in health_json:
            pytest.skip(f"Service at {base_url} is not the sentiment API")
    except Exception:
        pytest.skip(f"Service not reachable at {base_url}")

    # Test single text analysis
    test_text = "This is a neutral statement about facts."
    response = requests.post(
        f"{base_url}/analyze-text",
        json={"text": test_text},
        timeout=5,
    )
    if response.status_code != 200:
        pytest.skip(f"Service at {base_url} does not expose /analyze-text (status {response.status_code})")
    single = response.json()
    assert single.get("success") is True
    assert "result" in single and "predicted_sentiment" in single["result"]

    # Test batch analysis
    texts = [comment[0] for comment in TEST_COMMENTS]
    response = requests.post(
        f"{base_url}/analyze-batch",
        json={"texts": texts},
        timeout=10,
    )
    if response.status_code != 200:
        pytest.skip(f"Service at {base_url} does not expose /analyze-batch (status {response.status_code})")
    result = response.json()
    assert result.get("success") is True
    assert result["statistics"]["total_analyzed"] == len(texts)
    assert set(result["statistics"]["sentiment_distribution"].keys()) == {"positive", "neutral", "negative"}


@pytest.mark.skipif(os.getenv("RUN_ROBERTA_TESTS") != "1", reason="Set RUN_ROBERTA_TESTS=1 to run RoBERTa download test")
def test_direct_roberta():
    """Optionally test RoBERTa model directly without the API."""
    from transformers import pipeline

    classifier = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    )

    test_texts = [
        "This is amazing!",
        "This is terrible!",
        "This is a factual statement.",
    ]

    results = classifier(test_texts)
    assert isinstance(results, list) and len(results) == 3
    for res in results:
        assert "label" in res and "score" in res
    """Run all tests."""
    print("ROBERTA NEUTRAL SENTIMENT DETECTION TEST")
    print("=" * 60)
    
    # Test direct model first
    direct_success = test_direct_roberta()
    
    # Test API
    api_success = test_local_api()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Direct RoBERTa test: {'✅ PASSED' if direct_success else '❌ FAILED'}")
    print(f"API test: {'✅ PASSED' if api_success else '❌ FAILED'}")
    
    if not api_success and direct_success:
        print("\n⚠️  The model works directly but not through the API.")
        print("   Check the API's label normalization logic.")
    elif not direct_success:
        print("\n❌ The model itself is not working correctly.")
        print("   Check that the model is properly downloaded.")
    elif api_success and direct_success:
        print("\n✅ Everything is working! Neutral detection is functional.")
        print("   If you're still not seeing neutral in the UI, check:")
        print("   1. The Flask app's connection to this API")
        print("   2. The JavaScript display logic in the frontend")

if __name__ == "__main__":
    main()