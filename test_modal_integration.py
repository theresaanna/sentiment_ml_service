#!/usr/bin/env python3
"""
Test script to verify Modal ML service integration
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add app directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.ml_service_client import MLServiceClient


def test_modal_integration():
    """Test the Modal ML service integration"""
    print("Testing Modal ML Service Integration")
    print("=" * 50)
    
    # Check environment variables
    base_url = os.getenv("MODAL_ML_BASE_URL")
    if not base_url:
        print("❌ ERROR: MODAL_ML_BASE_URL not set in environment")
        return False
    
    print(f"✅ Modal URL configured: {base_url}")
    
    try:
        # Initialize the client
        client = MLServiceClient()
        print("✅ ML Service client initialized successfully")
        
        # Test single text analysis
        print("\n📝 Testing single text analysis...")
        test_text = "This is absolutely amazing! I love it!"
        result = client.analyze_text(test_text)
        
        print(f"   Text: '{test_text}'")
        print(f"   Sentiment: {result['predicted_sentiment']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   Method: {result['method']}")
        print(f"   Models used: {', '.join(result.get('models_used', []))}")
        
        # Test batch analysis
        print("\n📝 Testing batch analysis...")
        test_texts = [
            "This product is fantastic!",
            "I'm not happy with the service.",
            "It's okay, nothing special.",
            "Absolutely terrible experience!",
            "Best purchase I've ever made!"
        ]
        
        batch_result = client.analyze_batch(test_texts)
        
        print(f"   Total analyzed: {batch_result['statistics']['total_analyzed']}")
        print(f"   Method: {batch_result['method']}")
        print("\n   Distribution:")
        
        stats = batch_result['statistics']
        for sentiment, count in stats['sentiment_distribution'].items():
            percentage = stats['sentiment_percentages'][sentiment]
            print(f"     {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
        
        print(f"\n   Average confidence: {stats['average_confidence']:.2%}")
        
        print("\n   Individual results:")
        for i, result in enumerate(batch_result['results'], 1):
            print(f"     {i}. '{result['text'][:50]}...' -> {result['predicted_sentiment']} ({result['confidence']:.2%})")
        
        print("\n✅ All tests passed! Modal integration is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_modal_integration()
    sys.exit(0 if success else 1)