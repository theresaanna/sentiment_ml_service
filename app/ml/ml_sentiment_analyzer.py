"""
ML-Enhanced Sentiment Analyzer

Combines machine learning models with rule-based approaches for
robust sentiment analysis with continuous learning capabilities.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import joblib
import numpy as np
from app.science.sentiment_analyzer import SentimentAnalyzer
from app.ml.feedback_collector import feedback_manager


class MLSentimentAnalyzer:
    """Enhanced sentiment analyzer using ML models"""
    
    def __init__(self, model_path: Optional[str] = None, use_fallback: bool = True):
        """
        Initialize the ML sentiment analyzer
        
        Args:
            model_path: Path to trained model, defaults to latest
            use_fallback: Whether to use rule-based analyzer as fallback
        """
        self.model = None
        self.model_metadata = {}
        self.model_version = "unknown"
        
        # Load ML model
        if model_path is None:
            model_path = "models/latest_model.pkl"
        
        self.load_model(model_path)
        
        # Initialize fallback analyzer if requested
        self.fallback_analyzer = None
        if use_fallback:
            self.fallback_analyzer = SentimentAnalyzer()
        
        # Performance tracking
        self.predictions_made = 0
        self.feedback_received = 0
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a trained ML model
        
        Args:
            model_path: Path to the model file
            
        Returns:
            True if model loaded successfully
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            print(f"Warning: Model not found at {model_path}")
            return False
        
        try:
            self.model = joblib.load(model_path)
            
            # Load metadata
            metadata_path = model_path.with_suffix('') + '_metadata.json'
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                    self.model_version = self.model_metadata.get('training_date', 'unknown')
            
            print(f"Loaded model: {model_path}")
            print(f"Model accuracy: {self.model_metadata.get('test_accuracy', 0):.3f}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def analyze_sentiment(self, text: str, return_probabilities: bool = True) -> Dict[str, Any]:
        """
        Analyze sentiment of text using ML model
        
        Args:
            text: Text to analyze
            return_probabilities: Whether to return probability distribution
            
        Returns:
            Sentiment analysis results
        """
        if not text or not text.strip():
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'method': 'empty_text'
            }
        
        # Try ML model first
        if self.model is not None:
            try:
                result = self._ml_predict(text, return_probabilities)
                self.predictions_made += 1
                return result
            except Exception as e:
                print(f"ML prediction failed: {e}")
        
        # Fall back to rule-based if available
        if self.fallback_analyzer:
            rule_result = self.fallback_analyzer.analyze_sentiment(text)
            # Convert to our format
            return {
                'sentiment': rule_result['overall_sentiment'],
                'confidence': rule_result['confidence'],
                'method': 'rule_based_fallback',
                'subjectivity': rule_result.get('subjectivity', 0.5),
                'emotions': rule_result.get('emotions', {})
            }
        
        # Default response if all else fails
        return {
            'sentiment': 'neutral',
            'confidence': 0.0,
            'method': 'default'
        }
    
    def _ml_predict(self, text: str, return_probabilities: bool = True) -> Dict[str, Any]:
        """
        Make prediction using ML model
        
        Args:
            text: Text to analyze
            return_probabilities: Whether to return probability distribution
            
        Returns:
            Prediction results
        """
        # Clean text
        cleaned_text = self._clean_text(text)
        
        # Make prediction
        prediction = self.model.predict([cleaned_text])[0]
        
        # Map numeric prediction to label
        label_map = {-1: 'negative', 0: 'neutral', 1: 'positive'}
        sentiment = label_map.get(prediction, 'neutral')
        
        result = {
            'sentiment': sentiment,
            'method': 'ml_model',
            'model_version': self.model_version,
            'original_text': text,
            'cleaned_text': cleaned_text
        }
        
        # Get probabilities if available
        if return_probabilities and hasattr(self.model.named_steps['classifier'], 'predict_proba'):
            probabilities = self.model.named_steps['classifier'].predict_proba([cleaned_text])[0]
            
            # Map probabilities to sentiment labels
            # Assuming order: negative, neutral, positive
            result['probabilities'] = {
                'negative': float(probabilities[0]) if len(probabilities) > 0 else 0,
                'neutral': float(probabilities[1]) if len(probabilities) > 1 else 0,
                'positive': float(probabilities[2]) if len(probabilities) > 2 else 0
            }
            result['confidence'] = float(max(probabilities))
        else:
            # Estimate confidence based on prediction
            result['confidence'] = 0.7  # Default confidence for models without probability
        
        # Add additional features
        result.update(self._extract_features(text))
        
        return result
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text for ML processing
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extract additional features from text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of features
        """
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'exclamation_marks': text.count('!'),
            'question_marks': text.count('?'),
            'capital_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0
        }
        
        # Detect strong emotions
        strong_positive = ['love', 'amazing', 'excellent', 'fantastic', 'wonderful']
        strong_negative = ['hate', 'terrible', 'awful', 'horrible', 'disgusting']
        
        text_lower = text.lower()
        features['strong_positive_words'] = sum(1 for word in strong_positive if word in text_lower)
        features['strong_negative_words'] = sum(1 for word in strong_negative if word in text_lower)
        
        return features
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze multiple texts efficiently
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of analysis results
        """
        results = []
        
        # Use batch prediction if ML model is available
        if self.model is not None and len(texts) > 1:
            try:
                # Clean all texts
                cleaned_texts = [self._clean_text(text) for text in texts]
                
                # Batch predict
                predictions = self.model.predict(cleaned_texts)
                
                # Get probabilities if available
                probabilities = None
                if hasattr(self.model.named_steps['classifier'], 'predict_proba'):
                    probabilities = self.model.named_steps['classifier'].predict_proba(cleaned_texts)
                
                # Format results
                label_map = {-1: 'negative', 0: 'neutral', 1: 'positive'}
                
                for i, (text, pred) in enumerate(zip(texts, predictions)):
                    result = {
                        'sentiment': label_map.get(pred, 'neutral'),
                        'method': 'ml_model_batch',
                        'model_version': self.model_version,
                        'original_text': text
                    }
                    
                    if probabilities is not None:
                        prob_dist = probabilities[i]
                        result['probabilities'] = {
                            'negative': float(prob_dist[0]),
                            'neutral': float(prob_dist[1]),
                            'positive': float(prob_dist[2])
                        }
                        result['confidence'] = float(max(prob_dist))
                    
                    results.append(result)
                
                self.predictions_made += len(texts)
                return results
                
            except Exception as e:
                print(f"Batch prediction failed: {e}")
        
        # Fall back to individual predictions
        for text in texts:
            results.append(self.analyze_sentiment(text))
        
        return results
    
    def collect_feedback(self,
                        text: str,
                        original_prediction: str,
                        user_correction: str,
                        comment_id: Optional[str] = None,
                        video_id: Optional[str] = None,
                        confidence: Optional[int] = None) -> str:
        """
        Collect user feedback on a prediction
        
        Args:
            text: The analyzed text
            original_prediction: Model's original prediction
            user_correction: User's corrected label
            comment_id: Optional comment identifier
            video_id: Optional video identifier
            confidence: User's confidence (1-5)
            
        Returns:
            Feedback ID
        """
        # Get original confidence if available
        original_confidence = 0.5
        if hasattr(self, '_last_prediction_confidence'):
            original_confidence = self._last_prediction_confidence
        
        feedback_id = feedback_manager.collector.collect_feedback(
            video_id=video_id or 'unknown',
            comment_id=comment_id or f"text_{hash(text)}",
            comment_text=text,
            original_prediction=original_prediction,
            original_confidence=original_confidence,
            user_correction=user_correction,
            user_confidence=confidence,
            model_version=self.model_version
        )
        
        self.feedback_received += 1
        
        return feedback_id
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics
        
        Returns:
            Dictionary of performance metrics
        """
        stats = {
            'predictions_made': self.predictions_made,
            'feedback_received': self.feedback_received,
            'feedback_rate': self.feedback_received / self.predictions_made if self.predictions_made > 0 else 0,
            'model_version': self.model_version,
            'model_accuracy': self.model_metadata.get('test_accuracy', 0),
            'has_ml_model': self.model is not None,
            'has_fallback': self.fallback_analyzer is not None
        }
        
        # Add feedback summary if available
        feedback_summary = feedback_manager.collector.get_feedback_summary(days=7)
        stats['recent_feedback'] = feedback_summary['total_feedback']
        stats['ready_for_retraining'] = feedback_manager.should_retrain()
        
        return stats
    
    def explain_prediction(self, text: str) -> Dict[str, Any]:
        """
        Provide explanation for a sentiment prediction
        
        Args:
            text: Text that was analyzed
            
        Returns:
            Explanation dictionary
        """
        result = self.analyze_sentiment(text)
        
        explanation = {
            'prediction': result['sentiment'],
            'confidence': result.get('confidence', 0),
            'method': result.get('method', 'unknown'),
            'factors': []
        }
        
        # Add feature importance if using ML model
        if result.get('method') == 'ml_model' and self.model is not None:
            # Get feature contributions (simplified)
            features = self._extract_features(text)
            
            # Identify key factors
            if features['strong_positive_words'] > 0:
                explanation['factors'].append(f"Contains {features['strong_positive_words']} strong positive words")
            if features['strong_negative_words'] > 0:
                explanation['factors'].append(f"Contains {features['strong_negative_words']} strong negative words")
            if features['exclamation_marks'] > 2:
                explanation['factors'].append(f"High emotional intensity ({features['exclamation_marks']} exclamation marks)")
            if features['capital_ratio'] > 0.3:
                explanation['factors'].append(f"High capitalization ({features['capital_ratio']:.0%})")
            
            # Add probability distribution if available
            if 'probabilities' in result:
                explanation['probability_distribution'] = result['probabilities']
                
                # Identify why this sentiment was chosen
                probs = result['probabilities']
                max_prob = max(probs.values())
                if max_prob < 0.5:
                    explanation['factors'].append("Low confidence - mixed signals in text")
                elif max_prob > 0.8:
                    explanation['factors'].append("High confidence - clear sentiment signals")
        
        # Add rule-based explanation if using fallback
        elif result.get('method') == 'rule_based_fallback':
            if 'emotions' in result:
                top_emotion = max(result['emotions'].items(), key=lambda x: x[1])
                explanation['factors'].append(f"Primary emotion: {top_emotion[0]} ({top_emotion[1]:.2f})")
            if 'subjectivity' in result:
                explanation['factors'].append(f"Subjectivity: {result['subjectivity']:.2f}")
        
        return explanation


# Convenience function for easy access
def get_ml_analyzer(force_reload: bool = False) -> MLSentimentAnalyzer:
    """
    Get or create ML sentiment analyzer instance
    
    Args:
        force_reload: Whether to force reload the model
        
    Returns:
        MLSentimentAnalyzer instance
    """
    global _ml_analyzer_instance
    
    if force_reload or not hasattr(get_ml_analyzer, '_instance'):
        get_ml_analyzer._instance = MLSentimentAnalyzer()
    
    return get_ml_analyzer._instance
