#!/usr/bin/env python3
"""
Test script to verify Modal ML service integration
"""
import os
import sys
import pytest
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure the project root is on the path for imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import MLServiceClient from the correct module
from ml_service_client import MLServiceClient


def test_modal_integration():
    """Test the Modal ML service integration.
    Skips if MODAL_ML_BASE_URL is not configured.
    """
    base_url = os.getenv("MODAL_ML_BASE_URL")
    if not base_url:
        pytest.skip("MODAL_ML_BASE_URL not set")

    # Initialize the client
    client = MLServiceClient()

    # Test single text analysis
    test_text = "This is absolutely amazing! I love it!"
    result = client.analyze_text(test_text)

    assert isinstance(result, dict)
    assert result["predicted_sentiment"] in {"positive", "neutral", "negative"}
    assert 0.0 <= result["confidence"] <= 1.0

    # Test batch analysis
    test_texts = [
        "This product is fantastic!",
        "I'm not happy with the service.",
        "It's okay, nothing special.",
        "Absolutely terrible experience!",
        "Best purchase I've ever made!",
    ]

    batch_result = client.analyze_batch(test_texts)
    assert isinstance(batch_result, dict)

    stats = batch_result["statistics"]
    assert stats["total_analyzed"] == len(test_texts)
    dist = stats["sentiment_distribution"]
    assert sum(dist.values()) == len(test_texts)
    assert set(dist.keys()) == {"positive", "neutral", "negative"}


if __name__ == "__main__":
    success = test_modal_integration()
    sys.exit(0 if success else 1)
