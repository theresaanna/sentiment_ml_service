"""
Sentiment Analysis using ensemble of RoBERTa and Gradient Boosting.
"""
import os
import math
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import logging
from app.cache import cache

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Ensemble sentiment analyzer using RoBERTa and Gradient Boosting."""
    
    def __init__(self, batch_size: int = 32):
        """Initialize the sentiment analyzer with RoBERTa and GB models.
        
        Args:
            batch_size: Number of texts to process in each batch for efficiency
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        # Initialize RoBERTa model for social media sentiment
        self.roberta_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.roberta_tokenizer = None
        self.roberta_model = None
        
        # Gradient Boosting components
        self.gb_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.gb_model = None
        
        # Model weights for ensemble (prefer RoBERTa as requested)
        self.roberta_weight = 0.7  # Higher weight for RoBERTa
        self.gb_weight = 0.3
        
        # Sentiment labels
        self.sentiment_labels = ['negative', 'neutral', 'positive']
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load or initialize the ML models."""
        try:
            # Load RoBERTa model
            logger.info("Loading RoBERTa model for sentiment analysis...")
            self.roberta_tokenizer = AutoTokenizer.from_pretrained(self.roberta_model_name)
            self.roberta_model = AutoModelForSequenceClassification.from_pretrained(
                self.roberta_model_name
            )
            self.roberta_model.to(self.device)
            self.roberta_model.eval()
            
            # Load or train Gradient Boosting model
            gb_model_path = os.path.join(os.path.dirname(__file__), 'models', 'gb_model.pkl')
            gb_vectorizer_path = os.path.join(os.path.dirname(__file__), 'models', 'gb_vectorizer.pkl')
            
            if os.path.exists(gb_model_path) and os.path.exists(gb_vectorizer_path):
                logger.info("Loading pre-trained Gradient Boosting model...")
                with open(gb_model_path, 'rb') as f:
                    self.gb_model = pickle.load(f)
                with open(gb_vectorizer_path, 'rb') as f:
                    self.gb_vectorizer = pickle.load(f)
            else:
                logger.info("Gradient Boosting model not found. Will train on first batch of data.")
                self.gb_model = None
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def _train_gb_model(self, texts: List[str], labels: List[int]):
        """Train the Gradient Boosting model on labeled data."""
        try:
            logger.info("Training Gradient Boosting model...")
            
            # Vectorize texts
            X = self.gb_vectorizer.fit_transform(texts)
            
            # Train model
            self.gb_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            self.gb_model.fit(X, labels)
            
            # Save model
            os.makedirs(os.path.join(os.path.dirname(__file__), 'models'), exist_ok=True)
            
            with open(os.path.join(os.path.dirname(__file__), 'models', 'gb_model.pkl'), 'wb') as f:
                pickle.dump(self.gb_model, f)
            with open(os.path.join(os.path.dirname(__file__), 'models', 'gb_vectorizer.pkl'), 'wb') as f:
                pickle.dump(self.gb_vectorizer, f)
                
            logger.info("Gradient Boosting model trained and saved.")
            
        except Exception as e:
            logger.error(f"Error training GB model: {e}")
            self.gb_model = None
    
    def _analyze_with_roberta(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using RoBERTa model."""
        try:
            # Tokenize and prepare input
            inputs = self.roberta_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.roberta_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predictions = predictions.cpu().numpy()[0]
            
            # Map to sentiment scores
            return {
                'negative': float(predictions[0]),
                'neutral': float(predictions[1]),
                'positive': float(predictions[2])
            }
            
        except Exception as e:
            logger.error(f"Error in RoBERTa analysis: {e}")
            return {'negative': 0.33, 'neutral': 0.34, 'positive': 0.33}
    
    def _analyze_with_gb(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using Gradient Boosting model."""
        if self.gb_model is None:
            # Return equal probabilities if model not trained
            return {'negative': 0.33, 'neutral': 0.34, 'positive': 0.33}
        
        try:
            # Vectorize text
            X = self.gb_vectorizer.transform([text])
            
            # Get predictions
            proba = self.gb_model.predict_proba(X)[0]
            
            return {
                'negative': float(proba[0]),
                'neutral': float(proba[1]),
                'positive': float(proba[2])
            }
            
        except Exception as e:
            logger.error(f"Error in GB analysis: {e}")
            return {'negative': 0.33, 'neutral': 0.34, 'positive': 0.33}
    
    def calculate_agreement_confidence(self, roberta_scores: Dict[str, float], 
                                      gb_scores: Dict[str, float], 
                                      ensemble_scores: Dict[str, float]) -> float:
        """
        Calculate confidence based on model agreement.
        
        Higher agreement between models indicates higher confidence.
        
        Args:
            roberta_scores: Sentiment scores from RoBERTa model
            gb_scores: Sentiment scores from Gradient Boosting model
            ensemble_scores: Combined ensemble scores
            
        Returns:
            Confidence score between 0 and 1
        """
        # Calculate disagreement between models for each sentiment
        disagreement = sum(abs(roberta_scores[s] - gb_scores[s]) 
                          for s in self.sentiment_labels)
        
        # Convert disagreement to agreement (max disagreement is 2)
        agreement = 1 - (disagreement / 2)
        
        # Get the confidence of the ensemble prediction
        prediction_confidence = max(ensemble_scores.values())
        
        # Combine agreement and prediction confidence
        # Give more weight to prediction confidence (60%) vs agreement (40%)
        final_confidence = 0.6 * prediction_confidence + 0.4 * agreement
        
        return final_confidence
    
    def calculate_entropy_confidence(self, ensemble_scores: Dict[str, float]) -> float:
        """
        Calculate confidence using entropy.
        
        Lower entropy indicates higher confidence (more certain prediction).
        
        Args:
            ensemble_scores: Combined ensemble scores
            
        Returns:
            Confidence score between 0 and 1
        """
        # Calculate entropy of the prediction distribution
        entropy = -sum(p * math.log(p + 1e-8) for p in ensemble_scores.values() if p > 0)
        
        # Maximum possible entropy for uniform distribution
        max_entropy = math.log(len(self.sentiment_labels))
        
        # Normalize entropy to get confidence (1 - normalized_entropy)
        confidence = 1 - (entropy / max_entropy) if max_entropy > 0 else 1.0
        
        return confidence
    
    def calculate_margin_confidence(self, ensemble_scores: Dict[str, float]) -> float:
        """
        Calculate confidence based on the margin between top two predictions.
        
        Larger margin indicates higher confidence.
        
        Args:
            ensemble_scores: Combined ensemble scores
            
        Returns:
            Confidence score between 0 and 1
        """
        sorted_scores = sorted(ensemble_scores.values(), reverse=True)
        
        if len(sorted_scores) >= 2:
            # Margin between best and second best
            margin = sorted_scores[0] - sorted_scores[1]
        else:
            margin = sorted_scores[0] if sorted_scores else 0
        
        # Margin can be at most 1.0 (100% vs 0%)
        return margin
    
    def analyze_sentiment(self, text: str) -> Dict[str, any]:
        """
        Analyze sentiment of a single text using ensemble method.
        
        Returns:
            Dict with sentiment scores and predicted class with multiple confidence metrics
        """
        # Get predictions from both models
        roberta_scores = self._analyze_with_roberta(text)
        gb_scores = self._analyze_with_gb(text)
        
        # Compute weighted ensemble scores
        ensemble_scores = {}
        for sentiment in self.sentiment_labels:
            ensemble_scores[sentiment] = (
                self.roberta_weight * roberta_scores[sentiment] +
                self.gb_weight * gb_scores[sentiment]
            )
        
        # Normalize scores to sum to 1
        total = sum(ensemble_scores.values())
        if total > 0:
            ensemble_scores = {k: v/total for k, v in ensemble_scores.items()}
        
        # Get predicted sentiment
        predicted_sentiment = max(ensemble_scores, key=ensemble_scores.get)
        
        # Calculate multiple confidence metrics
        simple_confidence = ensemble_scores[predicted_sentiment]
        agreement_confidence = self.calculate_agreement_confidence(
            roberta_scores, gb_scores, ensemble_scores
        )
        entropy_confidence = self.calculate_entropy_confidence(ensemble_scores)
        margin_confidence = self.calculate_margin_confidence(ensemble_scores)
        
        # Calculate combined confidence (weighted average of all metrics)
        combined_confidence = (
            0.25 * simple_confidence +
            0.35 * agreement_confidence +
            0.25 * entropy_confidence +
            0.15 * margin_confidence
        )
        
        return {
            'text': text[:100] + '...' if len(text) > 100 else text,
            'ensemble_scores': ensemble_scores,
            'roberta_scores': roberta_scores,
            'gb_scores': gb_scores,
            'predicted_sentiment': predicted_sentiment,
            'confidence': combined_confidence,  # Use combined confidence as main metric
            'confidence_metrics': {
                'simple': simple_confidence,
                'agreement': agreement_confidence,
                'entropy': entropy_confidence,
                'margin': margin_confidence,
                'combined': combined_confidence
            }
        }
    
    def analyze_batch(self, texts: List[str], progress_callback=None) -> Dict[str, any]:
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts: List of text strings to analyze
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dict with overall statistics and individual results
        """
        results = []
        sentiment_counts = {'negative': 0, 'neutral': 0, 'positive': 0}
        
        # For large datasets, process in smaller batches to avoid memory issues
        if len(texts) > self.batch_size:
            return self.analyze_large_dataset(texts, progress_callback)
        
        # If GB model not trained and we have enough data, train it
        if self.gb_model is None and len(texts) > 50:
            # Use RoBERTa to label a subset for training GB
            sample_size = min(200, len(texts))
            sample_texts = texts[:sample_size]
            sample_labels = []
            
            for text in sample_texts:
                roberta_result = self._analyze_with_roberta(text)
                label = self.sentiment_labels.index(
                    max(roberta_result, key=roberta_result.get)
                )
                sample_labels.append(label)
            
            self._train_gb_model(sample_texts, sample_labels)
        
        # Analyze each text
        for i, text in enumerate(texts):
            if progress_callback:
                progress_callback(i + 1, len(texts))
            
            result = self.analyze_sentiment(text)
            results.append(result)
            sentiment_counts[result['predicted_sentiment']] += 1
        
        # Calculate overall statistics
        total_comments = len(texts)
        sentiment_percentages = {
            k: (v / total_comments * 100) if total_comments > 0 else 0
            for k, v in sentiment_counts.items()
        }
        
        # Calculate average confidence scores for all metrics
        if results:
            avg_confidence = np.mean([r['confidence'] for r in results])
            avg_confidence_metrics = {
                'simple': np.mean([r.get('confidence_metrics', {}).get('simple', 0) for r in results]),
                'agreement': np.mean([r.get('confidence_metrics', {}).get('agreement', 0) for r in results]),
                'entropy': np.mean([r.get('confidence_metrics', {}).get('entropy', 0) for r in results]),
                'margin': np.mean([r.get('confidence_metrics', {}).get('margin', 0) for r in results]),
                'combined': np.mean([r.get('confidence_metrics', {}).get('combined', r['confidence']) for r in results])
            }
        else:
            avg_confidence = 0
            avg_confidence_metrics = {
                'simple': 0,
                'agreement': 0,
                'entropy': 0,
                'margin': 0,
                'combined': 0
            }
        
        # Calculate sentiment score (-1 to 1 scale)
        sentiment_score = 0
        if total_comments > 0:
            sentiment_score = (
                (sentiment_counts['positive'] - sentiment_counts['negative']) / 
                total_comments
            )
        
        # Identify low confidence predictions for review
        confidence_threshold = 0.6
        low_confidence_results = [
            r for r in results 
            if r['confidence'] < confidence_threshold
        ]
        
        return {
            'total_analyzed': total_comments,
            'sentiment_counts': sentiment_counts,
            'sentiment_percentages': sentiment_percentages,
            'overall_sentiment': self._get_overall_sentiment(sentiment_score),
            'sentiment_score': sentiment_score,
            'average_confidence': avg_confidence,
            'average_confidence_metrics': avg_confidence_metrics,
            'low_confidence_count': len(low_confidence_results),
            'confidence_threshold': confidence_threshold,
            'individual_results': results,
            'model_weights': {
                'roberta': self.roberta_weight,
                'gradient_boosting': self.gb_weight
            }
        }
    
    def analyze_large_dataset(self, texts: List[str], progress_callback=None) -> Dict[str, any]:
        """
        Efficiently analyze large datasets by processing in batches.
        
        Args:
            texts: Large list of text strings to analyze
            progress_callback: Optional callback for progress updates
            
        Returns:
            Aggregated results from all batches
        """
        total_texts = len(texts)
        all_results = []
        sentiment_counts = {'negative': 0, 'neutral': 0, 'positive': 0}
        
        logger.info(f"Processing {total_texts} texts in batches of {self.batch_size}")
        
        # Process in batches
        for batch_start in range(0, total_texts, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_texts)
            batch_texts = texts[batch_start:batch_end]
            
            # Update progress
            if progress_callback:
                progress_callback(batch_end, total_texts)
            
            # Process batch with RoBERTa (more efficient for batches)
            batch_results = self._analyze_batch_with_roberta(batch_texts)
            
            # Process each result
            for i, text in enumerate(batch_texts):
                roberta_scores = batch_results[i]
                gb_scores = self._analyze_with_gb(text) if self.gb_model else roberta_scores
                
                # Compute ensemble scores
                ensemble_scores = {}
                for sentiment in self.sentiment_labels:
                    ensemble_scores[sentiment] = (
                        self.roberta_weight * roberta_scores[sentiment] +
                        self.gb_weight * gb_scores[sentiment]
                    )
                
                # Normalize
                total = sum(ensemble_scores.values())
                if total > 0:
                    ensemble_scores = {k: v/total for k, v in ensemble_scores.items()}
                
                predicted_sentiment = max(ensemble_scores, key=ensemble_scores.get)
                confidence = ensemble_scores[predicted_sentiment]
                
                result = {
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'ensemble_scores': ensemble_scores,
                    'predicted_sentiment': predicted_sentiment,
                    'confidence': confidence
                }
                
                all_results.append(result)
                sentiment_counts[predicted_sentiment] += 1
            
            # Clear GPU cache if using CUDA
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Calculate final statistics
        total_comments = len(texts)
        sentiment_percentages = {
            k: (v / total_comments * 100) if total_comments > 0 else 0
            for k, v in sentiment_counts.items()
        }
        
        avg_confidence = np.mean([r['confidence'] for r in all_results]) if all_results else 0
        
        sentiment_score = 0
        if total_comments > 0:
            sentiment_score = (
                (sentiment_counts['positive'] - sentiment_counts['negative']) / 
                total_comments
            )
        
        return {
            'total_analyzed': total_comments,
            'sentiment_counts': sentiment_counts,
            'sentiment_percentages': sentiment_percentages,
            'overall_sentiment': self._get_overall_sentiment(sentiment_score),
            'sentiment_score': sentiment_score,
            'average_confidence': avg_confidence,
            'individual_results': all_results,
            'batch_processing': True,
            'batch_size': self.batch_size
        }
    
    def _analyze_batch_with_roberta(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Efficiently analyze multiple texts with RoBERTa in a single batch.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment score dictionaries
        """
        try:
            # Tokenize all texts at once
            inputs = self.roberta_tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions for all texts
            with torch.no_grad():
                outputs = self.roberta_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predictions = predictions.cpu().numpy()
            
            # Convert to list of score dictionaries
            results = []
            for pred in predictions:
                results.append({
                    'negative': float(pred[0]),
                    'neutral': float(pred[1]),
                    'positive': float(pred[2])
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch RoBERTa analysis: {e}")
            # Return neutral scores for all texts on error
            return [{'negative': 0.33, 'neutral': 0.34, 'positive': 0.33} for _ in texts]
    
    def _get_overall_sentiment(self, score: float) -> str:
        """Get overall sentiment label from score."""
        if score < -0.3:
            return "Very Negative"
        elif score < -0.1:
            return "Negative"
        elif score < 0.1:
            return "Neutral"
        elif score < 0.3:
            return "Positive"
        else:
            return "Very Positive"
    
    def get_sentiment_timeline(self, comments: List[Dict]) -> List[Dict]:
        """
        Analyze sentiment over time for comments with timestamps.
        
        Args:
            comments: List of comment dicts with 'text' and 'published_at' fields
            
        Returns:
            List of sentiment data points over time
        """
        timeline = []
        
        for comment in comments:
            if 'text' not in comment:
                continue
                
            result = self.analyze_sentiment(comment['text'])
            timeline.append({
                'timestamp': comment.get('published_at', ''),
                'sentiment': result['predicted_sentiment'],
                'score': result['ensemble_scores'],
                'text_preview': result['text']
            })
        
        return timeline
