#!/usr/bin/env python3
"""
Test script to verify OpenAI CommentSummarizer functionality
"""
import os
import sys

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_openai_availability():
    """Test if OpenAI API key is available"""
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print("✅ OpenAI API key is available")
        print(f"   Key starts with: {api_key[:8]}...")
        return True
    else:
        print("❌ OpenAI API key is NOT available")
        print("   Please set the OPENAI_API_KEY environment variable")
        return False

def test_summarizer_import():
    """Test if CommentSummarizer can be imported"""
    try:
        from app_modules.science.comment_summarizer import CommentSummarizer
        print("✅ CommentSummarizer imported successfully")
        return CommentSummarizer
    except Exception as e:
        print(f"❌ Failed to import CommentSummarizer: {e}")
        return None

def test_summarizer_initialization():
    """Test if CommentSummarizer can be initialized"""
    try:
        from app_modules.science.comment_summarizer import CommentSummarizer
        summarizer = CommentSummarizer()
        print("✅ CommentSummarizer initialized successfully")
        return summarizer
    except Exception as e:
        print(f"❌ Failed to initialize CommentSummarizer: {e}")
        return None

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
    
    try:
        result = summarizer.generate_summary(sample_comments, sample_sentiment)
        print("✅ Sample summarization successful!")
        print(f"   Method used: {result.get('method', 'unknown')}")
        print(f"   Comments analyzed: {result.get('comments_analyzed', 0)}")
        print(f"   Summary length: {len(result.get('summary', '').split())} words")
        print("\n📝 Generated Summary:")
        print("-" * 50)
        print(result.get('summary', 'No summary generated'))
        print("-" * 50)
        return True
    except Exception as e:
        print(f"❌ Sample summarization failed: {e}")
        return False

def main():
    print("🔍 Testing OpenAI CommentSummarizer Setup")
    print("=" * 50)
    
    # Test 1: Check OpenAI API key
    has_key = test_openai_availability()
    print()
    
    # Test 2: Import summarizer
    summarizer_class = test_summarizer_import()
    if not summarizer_class:
        return
    print()
    
    # Test 3: Initialize summarizer (this will fail if no API key)
    if not has_key:
        print("⚠️  Cannot test summarizer initialization without API key")
        print("   Please run: export OPENAI_API_KEY='your-api-key-here'")
        return
        
    summarizer = test_summarizer_initialization()
    if not summarizer:
        return
    print()
    
    # Test 4: Try sample summarization
    test_sample_summarization(summarizer)
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    main()