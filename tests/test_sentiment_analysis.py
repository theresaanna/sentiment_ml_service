"""
Comprehensive tests for the sentiment analysis service.
"""
import pytest
import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from fastapi.testclient import TestClient
from app import app, get_pipeline

client = TestClient(app)


class TestSentimentAnalysis:
    """Test suite for sentiment analysis endpoints."""
    
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model" in data
        assert data["model"] == "distilbert-base-uncased-finetuned-sst-2-english"
    
    def test_analyze_positive_text(self):
        """Test analyzing a positive sentiment text."""
        response = client.post(
            "/analyze-text",
            json={"text": "I absolutely love this product! It's amazing and wonderful!"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["result"]["predicted_sentiment"] == "positive"
        assert data["result"]["confidence"] > 0.8
        assert "distilbert" in data["models_used"]
    
    def test_analyze_negative_text(self):
        """Test analyzing a negative sentiment text."""
        response = client.post(
            "/analyze-text",
            json={"text": "This is terrible. I hate it. Very disappointed."}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["result"]["predicted_sentiment"] == "negative"
        assert data["result"]["confidence"] > 0.7
    
    def test_analyze_neutral_text(self):
        """Test analyzing a neutral sentiment text."""
        response = client.post(
            "/analyze-text",
            json={"text": "The weather today is cloudy."}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        # Note: Model might classify as positive/negative, but with lower confidence
        assert data["result"]["predicted_sentiment"] in ["positive", "negative", "neutral"]
        assert "confidence" in data["result"]
    
    def test_analyze_batch(self):
        """Test batch analysis of multiple texts."""
        texts = [
            "I love this!",
            "This is terrible.",
            "It's okay, I guess.",
            "Amazing product, highly recommend!",
            "Worst experience ever."
        ]
        response = client.post(
            "/analyze-batch",
            json={"texts": texts}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["results"]) == 5
        assert data["statistics"]["total_analyzed"] == 5
        
        # Check sentiment distribution
        distribution = data["statistics"]["sentiment_distribution"]
        assert distribution["positive"] >= 2
        assert distribution["negative"] >= 2
        assert "overall_sentiment" in data
    
    def test_analyze_batch_empty(self):
        """Test batch analysis with empty input."""
        response = client.post(
            "/analyze-batch",
            json={"texts": []}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["results"] == []
        assert data["statistics"]["total_analyzed"] == 0
    
    def test_analyze_batch_long_texts(self):
        """Test batch analysis with long texts (should truncate)."""
        long_text = "This is amazing! " * 100  # Very long text
        response = client.post(
            "/analyze-batch",
            json={"texts": [long_text]}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["results"]) == 1
        # Check that text is truncated in results
        assert len(data["results"][0]["text"]) == 100
    
    def test_analyze_text_with_special_characters(self):
        """Test analyzing text with special characters and emojis."""
        response = client.post(
            "/analyze-text",
            json={"text": "I ❤️ this! 😊 #amazing @product"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["result"]["predicted_sentiment"] == "positive"
    
    def test_invalid_request_missing_text(self):
        """Test handling of invalid request without text field."""
        response = client.post(
            "/analyze-text",
            json={}
        )
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_invalid_request_missing_texts(self):
        """Test handling of invalid batch request without texts field."""
        response = client.post(
            "/analyze-batch",
            json={}
        )
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_model_loading(self):
        """Test that the model loads correctly."""
        pipeline = get_pipeline()
        assert pipeline is not None
        
        # Test a simple prediction
        result = pipeline(["Test text"])[0]
        assert "label" in result
        assert "score" in result


class TestPerformance:
    """Performance-related tests."""
    
    def test_batch_performance(self):
        """Test that batch processing is efficient."""
        import time
        
        # Test with 10 texts
        texts = ["This is test text number " + str(i) for i in range(10)]
        
        start_time = time.time()
        response = client.post(
            "/analyze-batch",
            json={"texts": texts}
        )
        end_time = time.time()
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 10
        
        # Should process 10 texts in under 5 seconds
        processing_time = end_time - start_time
        assert processing_time < 5.0, f"Batch processing took {processing_time} seconds"
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        import concurrent.futures
        
        def make_request(text):
            return client.post(
                "/analyze-text",
                json={"text": text}
            )
        
        texts = [f"Test text {i}" for i in range(5)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, text) for text in texts]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        for response in results:
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])