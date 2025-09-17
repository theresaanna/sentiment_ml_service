#!/usr/bin/env python3
"""
Test script to verify that the RoBERTa model correctly detects neutral sentiment.
This will help diagnose why neutral comments aren't showing up in the UI.
"""

import requests
import json
from typing import List, Dict

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

def test_local_api(base_url: str = "http://localhost:8000"):
    """Test the local FastAPI sentiment service."""
    print("=" * 60)
    print("Testing Local FastAPI Service")
    print(f"URL: {base_url}")
    print("=" * 60)
    
    # Test single text analysis
    print("\n1. Testing single text analysis:")
    test_text = "This is a neutral statement about facts."
    
    try:
        response = requests.post(
            f"{base_url}/analyze-text",
            json={"text": test_text}
        )
        result = response.json()
        print(f"Text: '{test_text}'")
        print(f"Result: {json.dumps(result, indent=2)}")
    except Exception as e:
        print(f"Error testing single text: {e}")
        return False
    
    # Test batch analysis
    print("\n2. Testing batch analysis with various sentiments:")
    texts = [comment[0] for comment in TEST_COMMENTS]
    
    try:
        response = requests.post(
            f"{base_url}/analyze-batch",
            json={"texts": texts}
        )
        result = response.json()
        
        if result.get("success"):
            # Count sentiments
            sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
            
            print("\nIndividual Results:")
            print("-" * 50)
            for i, item in enumerate(result.get("results", [])):
                expected = TEST_COMMENTS[i][1]
                predicted = item.get("predicted_sentiment", "unknown")
                confidence = item.get("confidence", 0)
                
                # Mark if prediction matches expectation
                match = "✓" if predicted == expected else "✗"
                
                print(f"{match} Text: '{TEST_COMMENTS[i][0][:50]}...'")
                print(f"  Expected: {expected}, Got: {predicted} (conf: {confidence:.2f})")
                
                sentiment_counts[predicted] = sentiment_counts.get(predicted, 0) + 1
            
            print("\n" + "=" * 50)
            print("Summary Statistics:")
            print(f"Total analyzed: {result['statistics']['total_analyzed']}")
            print(f"Distribution: {result['statistics']['sentiment_distribution']}")
            print(f"Percentages: {result['statistics']['sentiment_percentages']}")
            
            # Check if neutral detection is working
            neutral_count = sentiment_counts.get("neutral", 0)
            neutral_expected = sum(1 for _, expected in TEST_COMMENTS if expected == "neutral")
            
            print("\n" + "=" * 50)
            print("NEUTRAL DETECTION TEST:")
            print(f"Expected neutral comments: {neutral_expected}")
            print(f"Detected neutral comments: {neutral_count}")
            
            if neutral_count == 0:
                print("❌ PROBLEM: No neutral comments detected!")
                print("   The model is not detecting ANY neutral sentiment.")
                return False
            elif neutral_count < neutral_expected * 0.5:
                print("⚠️  WARNING: Less than 50% of neutral comments detected")
                return False
            else:
                print("✅ SUCCESS: Neutral detection is working!")
                return True
                
        else:
            print(f"API returned error: {result}")
            return False
            
    except Exception as e:
        print(f"Error testing batch: {e}")
        return False

def test_direct_roberta():
    """Test RoBERTa model directly without the API."""
    print("\n" + "=" * 60)
    print("Testing RoBERTa Model Directly")
    print("=" * 60)
    
    try:
        from transformers import pipeline
        
        # Load the exact same model
        print("Loading model: cardiffnlp/twitter-roberta-base-sentiment-latest")
        classifier = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        
        # Test a few examples
        test_texts = [
            "This is amazing!",
            "This is terrible!",
            "This is a factual statement.",
            "The video is 10 minutes long.",
            "What time is it?",
        ]
        
        print("\nDirect model predictions:")
        print("-" * 40)
        for text in test_texts:
            result = classifier(text)[0]
            label = result['label']
            score = result['score']
            
            # Map label
            if label == 'LABEL_0':
                sentiment = 'negative'
            elif label == 'LABEL_1':
                sentiment = 'neutral'
            elif label == 'LABEL_2':
                sentiment = 'positive'
            else:
                sentiment = label.lower()
            
            print(f"Text: '{text}'")
            print(f"  Raw label: {label} -> Sentiment: {sentiment} (confidence: {score:.3f})")
        
        return True
        
    except Exception as e:
        print(f"Error testing direct model: {e}")
        return False

def main():
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