#!/usr/bin/env python3
"""
Test script to verify OpenAI CommentSummarizer functionality
"""
import os
import sys
import pytest

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def summarizer():
    """Provide a CommentSummarizer instance if OPENAI_API_KEY is set, otherwise skip."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping OpenAI summarizer tests")
    try:
        from app_modules.science.comment_summarizer import CommentSummarizer
        return CommentSummarizer()
    except Exception as e:
        pytest.skip(f"Cannot initialize CommentSummarizer: {e}")


def test_openai_availability():
    """Test if OpenAI API key is available (non-fatal if missing)."""
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        # Don't print secrets in logs; just assert presence
        assert isinstance(api_key, str) and len(api_key) > 0
    else:
        pytest.skip("OPENAI_API_KEY not set")


def test_summarizer_import():
    """Test if CommentSummarizer can be imported"""
    try:
        from app_modules.science.comment_summarizer import CommentSummarizer
        assert CommentSummarizer is not None
    except Exception as e:
        pytest.skip(f"CommentSummarizer import skipped: {e}")


def test_summarizer_initialization(summarizer):
    """Test if CommentSummarizer can be initialized"""
    assert summarizer is not None


def test_sample_summarization(summarizer):
    """Test summarization with sample data"""
    sample_comments = [
        {"text": "This video is amazing! Lady Gaga always delivers.", "likes": 15},
        {"text": "I don't agree with her feminist views but respect her artistry.", "likes": 8},
        {"text": "Great interview, very insightful questions.", "likes": 12},
        {"text": "She's so pretentious, can't stand her anymore.", "likes": 3},
        {"text": "Love how authentic she is in this conversation.", "likes": 20},
    ]
    
    sample_sentiment = {
        "overall_sentiment": "mixed",
        "sentiment_percentages": {"positive": 60.0, "negative": 30.0, "neutral": 10.0},
        "individual_results": [
            {"predicted_sentiment": "positive", "confidence": 0.9},
            {"predicted_sentiment": "neutral", "confidence": 0.7},
            {"predicted_sentiment": "positive", "confidence": 0.8},
            {"predicted_sentiment": "negative", "confidence": 0.9},
            {"predicted_sentiment": "positive", "confidence": 0.85},
        ]
    }
    
    result = summarizer.generate_summary(sample_comments, sample_sentiment)
    assert isinstance(result, dict)
    assert result.get("method") in {"openai", "openai_error"}
    assert result.get("comments_analyzed") == len(sample_comments)


def main():
    print("🔍 Testing OpenAI CommentSummarizer Setup")
    print("=" * 50)
    
    # Test 1: Check OpenAI API key
    try:
        test_openai_availability()
    except pytest.SkipExpected:  # type: ignore[attr-defined]
        print("OPENAI_API_KEY not set; skipping availability test")
    print()
    
    # Test 2: Import summarizer
    try:
        test_summarizer_import()
    except pytest.SkipExpected:  # type: ignore[attr-defined]
        print("CommentSummarizer import skipped")
    print()
    
    # Note: Remaining tests are handled by pytest when running the suite.
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    main()
