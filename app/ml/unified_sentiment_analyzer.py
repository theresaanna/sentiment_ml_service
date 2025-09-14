"""
Unified Sentiment Analyzer with Feedback Integration

This module combines all sentiment analysis approaches (RoBERTa, Gradient Boosting, 
Fast DistilBERT, and ML models) into a single, coherent system with continuous 
learning through feedback integration.
"""

import os
import json
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

# Import all analyzer components
from app.science.sentiment_analyzer import SentimentAnalyzer
from app.science.fast_sentiment_analyzer import FastSentimentAnalyzer
from app.ml.ml_sentiment_analyzer import MLSentimentAnalyzer
from app.ml.feedback_collector import FeedbackCollector
from app.ml.model_trainer import SentimentModelTrainer
from app.cache import cache

logger = logging.getLogger(__name__)


class UnifiedSentimentAnalyzer:
    """
    Unified sentiment analyzer that intelligently combines multiple approaches
    and continuously improves through user feedback.
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 enable_feedback: bool = True,
                 auto_retrain: bool = True):
        """
        Initialize the unified sentiment analyzer.
        
        Args:
            config: Configuration dictionary for customization
            enable_feedback: Whether to enable feedback collection
            auto_retrain: Whether to automatically retrain models with feedback
        """
        self.config = config or self._get_default_config()
        self.enable_feedback = enable_feedback
        self.auto_retrain = auto_retrain
        
        # Initialize all analyzer components
        self.analyzers = {}
        self._initialize_analyzers()
        
        # Initialize feedback system
        self.feedback_collector = None
        self.model_trainer = None
        if enable_feedback:
            self.feedback_collector = FeedbackCollector()
            self.model_trainer = SentimentModelTrainer()
        
        # Performance tracking
        self.performance_metrics = {
            'total_analyses': 0,
            'analyzer_usage': {},
            'feedback_collected': 0,
            'model_retrained_count': 0,
            'last_retrain': None
        }
        
        # Model weights for ensemble (adaptive based on performance)
        self.model_weights = self.config['model_weights'].copy()
        
        logger.info("UnifiedSentimentAnalyzer initialized with feedback=%s, auto_retrain=%s", 
                   enable_feedback, auto_retrain)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'analyzers': {
                'roberta': {'enabled': True, 'batch_size': 32},
                'fast': {'enabled': True, 'batch_size': 32},
                'ml': {'enabled': True, 'use_fallback': False},
                'ensemble': {'enabled': True}
            },
            'model_weights': {
                'roberta': 0.4,
                'fast': 0.3,
                'ml': 0.3
            },
            'feedback': {
                'min_confidence_for_auto_accept': 0.9,
                'feedback_batch_size': 100,
                'retrain_threshold': 500  # Retrain after N feedback items
            },
            'performance': {
                'cache_ttl_hours': 6,
                'max_batch_size': 100,
                'enable_gpu': torch.cuda.is_available()
            }
        }
    
    def _initialize_analyzers(self):
        """Initialize all enabled analyzer components."""
        config = self.config['analyzers']
        
        try:
            # Initialize RoBERTa-based analyzer
            if config['roberta']['enabled']:
                self.analyzers['roberta'] = SentimentAnalyzer(
                    batch_size=config['roberta']['batch_size']
                )
                logger.info("RoBERTa analyzer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize RoBERTa analyzer: {e}")
        
        try:
            # Initialize Fast DistilBERT analyzer
            if config['fast']['enabled']:
                self.analyzers['fast'] = FastSentimentAnalyzer(
                    batch_size=config['fast']['batch_size']
                )
                logger.info("Fast analyzer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Fast analyzer: {e}")
        
        try:
            # Initialize ML analyzer with trained models
            if config['ml']['enabled']:
                self.analyzers['ml'] = MLSentimentAnalyzer(
                    use_fallback=config['ml']['use_fallback']
                )
                logger.info("ML analyzer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize ML analyzer: {e}")
    
    def analyze_sentiment(self, 
                         text: str, 
                         method: str = 'auto',
                         collect_feedback: bool = True,
                         context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze sentiment of a single text using specified or automatic method.
        
        Args:
            text: Text to analyze
            method: Analysis method ('auto', 'ensemble', 'roberta', 'fast', 'ml')
            collect_feedback: Whether to enable feedback collection for this analysis
            context: Optional context (video_id, comment_id, etc.)
            
        Returns:
            Comprehensive sentiment analysis results
        """
        if not text or not text.strip():
            return self._empty_result()
        
        start_time = datetime.now()
        
        # Determine analysis method
        if method == 'auto':
            method = self._select_best_method(text, context)
        
        # Perform analysis
        if method == 'ensemble':
            result = self._ensemble_analysis(text)
        else:
            result = self._single_analyzer_analysis(text, method)
        
        # Add metadata
        result['metadata'] = {
            'analysis_method': method,
            'analysis_time': (datetime.now() - start_time).total_seconds(),
            'text_length': len(text),
            'feedback_enabled': collect_feedback and self.enable_feedback
        }
        
        # Add context if provided
        if context:
            result['context'] = context
        
        # Track performance
        self._update_performance_metrics(method)
        
        # Store for potential feedback
        if collect_feedback and self.enable_feedback:
            result['feedback_id'] = self._prepare_for_feedback(result)
        
        return result
    
    def analyze_batch(self,
                     texts: List[str],
                     method: str = 'auto',
                     batch_size: Optional[int] = None,
                     progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Analyze sentiment for a batch of texts efficiently.
        
        Args:
            texts: List of texts to analyze
            method: Analysis method
            batch_size: Batch size for processing
            progress_callback: Optional callback for progress updates
            
        Returns:
            Batch analysis results with statistics
        """
        if not texts:
            return {'results': [], 'statistics': {}}
        
        batch_size = batch_size or self.config['performance']['max_batch_size']
        total_texts = len(texts)
        
        # Determine best method for batch
        if method == 'auto':
            # Use fast analyzer for large batches
            method = 'fast' if total_texts > 100 else 'ensemble'
        
        logger.info(f"Batch analyzing {total_texts} texts using {method} method")
        
        # Process in batches
        all_results = []
        for i in range(0, total_texts, batch_size):
            batch = texts[i:i + batch_size]
            
            if progress_callback:
                progress_callback(i + len(batch), total_texts)
            
            # Analyze batch
            if method == 'ensemble':
                batch_results = self._batch_ensemble_analysis(batch)
            elif method in self.analyzers:
                batch_results = self._batch_single_analyzer(batch, method)
            else:
                batch_results = [self._empty_result() for _ in batch]
            
            all_results.extend(batch_results)
        
        # Calculate statistics
        statistics = self._calculate_batch_statistics(all_results)
        
        return {
            'results': all_results,
            'statistics': statistics,
            'method': method,
            'total_analyzed': total_texts
        }
    
    def _ensemble_analysis(self, text: str) -> Dict[str, Any]:
        """
        Perform ensemble analysis using all available analyzers.
        
        Args:
            text: Text to analyze
            
        Returns:
            Ensemble analysis results
        """
        individual_results = {}
        
        # Get predictions from each analyzer
        for name, analyzer in self.analyzers.items():
            try:
                if name == 'roberta':
                    result = analyzer.analyze_sentiment(text)
                    individual_results[name] = {
                        'sentiment': result['predicted_sentiment'],
                        'confidence': result['confidence'],
                        'scores': result.get('ensemble_scores', {})
                    }
                elif name == 'fast':
                    # Use fast analyzer's single text method
                    texts = [text]
                    result = analyzer.analyze_batch_fast(texts)
                    if result['individual_results']:
                        ind_result = result['individual_results'][0]
                        individual_results[name] = {
                            'sentiment': ind_result['predicted_sentiment'],
                            'confidence': ind_result['confidence'],
                            'scores': ind_result.get('sentiment_scores', {})
                        }
                elif name == 'ml':
                    result = analyzer.analyze_sentiment(text)
                    individual_results[name] = {
                        'sentiment': result['sentiment'],
                        'confidence': result['confidence'],
                        'scores': result.get('probabilities', {})
                    }
            except Exception as e:
                logger.warning(f"Analyzer {name} failed: {e}")
                continue
        
        # Combine results using weighted voting
        ensemble_result = self._weighted_ensemble_voting(individual_results)
        ensemble_result['individual_predictions'] = individual_results
        
        return ensemble_result
    
    def _weighted_ensemble_voting(self, individual_results: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Combine individual analyzer results using weighted voting.
        
        Args:
            individual_results: Results from each analyzer
            
        Returns:
            Combined ensemble result
        """
        if not individual_results:
            return self._empty_result()
        
        # Initialize sentiment scores
        sentiment_scores = {'positive': 0, 'neutral': 0, 'negative': 0}
        total_weight = 0
        
        # Aggregate weighted scores
        for analyzer_name, result in individual_results.items():
            weight = self.model_weights.get(analyzer_name, 0.33)
            confidence = result.get('confidence', 0.5)
            
            # Add to sentiment scores
            sentiment = result.get('sentiment', 'neutral')
            if sentiment in sentiment_scores:
                sentiment_scores[sentiment] += weight * confidence
            
            # If we have probability distribution, use it
            if 'scores' in result and result['scores']:
                for sent, score in result['scores'].items():
                    if sent in sentiment_scores:
                        sentiment_scores[sent] += weight * score * 0.5  # Blend with direct prediction
            
            total_weight += weight
        
        # Normalize scores
        if total_weight > 0:
            sentiment_scores = {k: v / total_weight for k, v in sentiment_scores.items()}
        
        # Get final prediction
        predicted_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        confidence = sentiment_scores[predicted_sentiment]
        
        # Calculate agreement score
        agreement = self._calculate_agreement(individual_results)
        
        return {
            'predicted_sentiment': predicted_sentiment,
            'confidence': confidence,
            'sentiment_scores': sentiment_scores,
            'agreement_score': agreement,
            'ensemble_method': 'weighted_voting',
            'analyzers_used': list(individual_results.keys())
        }
    
    def _calculate_agreement(self, individual_results: Dict[str, Dict]) -> float:
        """Calculate agreement score between analyzers."""
        if len(individual_results) < 2:
            return 1.0
        
        sentiments = [r['sentiment'] for r in individual_results.values()]
        most_common = max(set(sentiments), key=sentiments.count)
        agreement = sentiments.count(most_common) / len(sentiments)
        
        return agreement
    
    def collect_user_feedback(self,
                             analysis_id: str,
                             correct_sentiment: str,
                             confidence: int = 4,
                             notes: Optional[str] = None) -> bool:
        """
        Collect user feedback on a sentiment prediction.
        
        Args:
            analysis_id: ID of the analysis to provide feedback for
            correct_sentiment: The correct sentiment according to user
            confidence: User's confidence in correction (1-5)
            notes: Optional notes about the correction
            
        Returns:
            True if feedback was collected successfully
        """
        if not self.enable_feedback or not self.feedback_collector:
            logger.warning("Feedback collection is disabled")
            return False
        
        try:
            # Retrieve original analysis from cache
            cached_analysis = cache.get('analysis_for_feedback', analysis_id)
            if not cached_analysis:
                logger.warning(f"Analysis {analysis_id} not found for feedback")
                return False
            
            # Collect feedback
            feedback_id = self.feedback_collector.collect_feedback(
                video_id=cached_analysis.get('context', {}).get('video_id', 'unknown'),
                comment_id=cached_analysis.get('context', {}).get('comment_id', analysis_id),
                comment_text=cached_analysis.get('text', ''),
                original_prediction=cached_analysis['predicted_sentiment'],
                original_confidence=cached_analysis['confidence'],
                user_correction=correct_sentiment,
                user_confidence=confidence,
                model_version=cached_analysis.get('metadata', {}).get('analysis_method', 'unknown'),
                additional_notes=notes
            )
            
            self.performance_metrics['feedback_collected'] += 1
            
            # Check if we should retrain
            if self.auto_retrain:
                self._check_and_retrain()
            
            logger.info(f"Feedback collected: {feedback_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to collect feedback: {e}")
            return False
    
    def _check_and_retrain(self):
        """Check if models should be retrained based on feedback count."""
        feedback_threshold = self.config['feedback']['retrain_threshold']
        
        if self.performance_metrics['feedback_collected'] >= feedback_threshold:
            logger.info("Feedback threshold reached, initiating retraining...")
            self.retrain_with_feedback()
    
    def retrain_with_feedback(self, algorithm: str = 'logistic_regression') -> Dict[str, Any]:
        """
        Retrain ML models using collected feedback.
        
        Args:
            algorithm: ML algorithm to use for training
            
        Returns:
            Training results and metrics
        """
        if not self.model_trainer or not self.feedback_collector:
            logger.error("Training components not initialized")
            return {'error': 'Training not available'}
        
        try:
            # Export feedback for training
            training_file = self.feedback_collector.export_for_training()
            
            # Load training data
            df, skipped = self.model_trainer.load_training_data([training_file])
            
            if len(df) < 100:
                logger.warning(f"Insufficient training data: {len(df)} samples")
                return {'error': 'Insufficient training data'}
            
            # Prepare features
            features = self.model_trainer.prepare_features(df)
            
            # Train model
            results = self.model_trainer.train_model(features, algorithm=algorithm)
            
            # Save model
            model_path = self.model_trainer.save_model(
                name=f"feedback_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Reload ML analyzer with new model
            if 'ml' in self.analyzers:
                self.analyzers['ml'].load_model(model_path)
            
            # Update metrics
            self.performance_metrics['model_retrained_count'] += 1
            self.performance_metrics['last_retrain'] = datetime.now().isoformat()
            self.performance_metrics['feedback_collected'] = 0  # Reset counter
            
            logger.info(f"Model retrained successfully: {model_path}")
            
            return {
                'success': True,
                'model_path': str(model_path),
                'metrics': results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            return {'error': str(e)}
    
    def _select_best_method(self, text: str, context: Optional[Dict] = None) -> str:
        """
        Automatically select the best analysis method based on text and context.
        
        Args:
            text: Text to analyze
            context: Optional context information
            
        Returns:
            Selected method name
        """
        text_length = len(text)
        
        # For very short texts, use fast analyzer
        if text_length < 50:
            return 'fast' if 'fast' in self.analyzers else 'ensemble'
        
        # For long texts, use RoBERTa for better context understanding
        if text_length > 500:
            return 'roberta' if 'roberta' in self.analyzers else 'ensemble'
        
        # If we have a trained ML model with good performance, prefer it
        if 'ml' in self.analyzers and self.performance_metrics.get('ml_accuracy', 0) > 0.85:
            return 'ml'
        
        # Default to ensemble for best accuracy
        return 'ensemble'
    
    def _single_analyzer_analysis(self, text: str, method: str) -> Dict[str, Any]:
        """Perform analysis using a single specified analyzer."""
        if method not in self.analyzers:
            logger.warning(f"Analyzer {method} not available")
            return self._empty_result()
        
        try:
            analyzer = self.analyzers[method]
            
            if method == 'roberta':
                result = analyzer.analyze_sentiment(text)
                return {
                    'predicted_sentiment': result['predicted_sentiment'],
                    'confidence': result['confidence'],
                    'sentiment_scores': result.get('ensemble_scores', {}),
                    'method': method
                }
            elif method == 'fast':
                results = analyzer.analyze_batch_fast([text])
                if results['individual_results']:
                    ind_result = results['individual_results'][0]
                    return {
                        'predicted_sentiment': ind_result['predicted_sentiment'],
                        'confidence': ind_result['confidence'],
                        'sentiment_scores': ind_result.get('sentiment_scores', {}),
                        'method': method
                    }
            elif method == 'ml':
                result = analyzer.analyze_sentiment(text)
                return {
                    'predicted_sentiment': result['sentiment'],
                    'confidence': result['confidence'],
                    'sentiment_scores': result.get('probabilities', {}),
                    'method': method
                }
        except Exception as e:
            logger.error(f"Analysis failed with {method}: {e}")
        
        return self._empty_result()
    
    def _batch_ensemble_analysis(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Perform ensemble analysis on a batch of texts."""
        batch_results = []
        
        for text in texts:
            result = self._ensemble_analysis(text)
            result['text'] = text[:100] + '...' if len(text) > 100 else text
            batch_results.append(result)
        
        return batch_results
    
    def _batch_single_analyzer(self, texts: List[str], method: str) -> List[Dict[str, Any]]:
        """Perform batch analysis using a single analyzer."""
        if method not in self.analyzers:
            return [self._empty_result() for _ in texts]
        
        analyzer = self.analyzers[method]
        
        try:
            if method in ['roberta', 'fast']:
                # These analyzers have batch methods
                if method == 'roberta':
                    results = analyzer.analyze_batch(texts)
                    return results['individual_results']
                else:  # fast
                    results = analyzer.analyze_batch_fast(texts)
                    return results['individual_results']
            else:
                # Process individually
                return [self._single_analyzer_analysis(text, method) for text in texts]
        except Exception as e:
            logger.error(f"Batch analysis failed with {method}: {e}")
            return [self._empty_result() for _ in texts]
    
    def _calculate_batch_statistics(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate statistics for batch analysis results."""
        if not results:
            return {}
        
        sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
        confidences = []
        agreements = []
        
        for result in results:
            sentiment = result.get('predicted_sentiment', 'neutral')
            sentiment_counts[sentiment] += 1
            
            confidence = result.get('confidence', 0)
            confidences.append(confidence)
            
            agreement = result.get('agreement_score', 1.0)
            agreements.append(agreement)
        
        total = len(results)
        
        return {
            'total_analyzed': total,
            'sentiment_distribution': sentiment_counts,
            'sentiment_percentages': {k: v/total*100 for k, v in sentiment_counts.items()},
            'average_confidence': np.mean(confidences) if confidences else 0,
            'average_agreement': np.mean(agreements) if agreements else 1.0,
            'confidence_std': np.std(confidences) if confidences else 0,
            'high_confidence_count': sum(1 for c in confidences if c > 0.8),
            'low_confidence_count': sum(1 for c in confidences if c < 0.5)
        }
    
    def _prepare_for_feedback(self, result: Dict[str, Any]) -> str:
        """Prepare analysis result for potential feedback collection."""
        import uuid
        analysis_id = str(uuid.uuid4())[:8]
        
        # Cache the analysis for later feedback
        cache.set('analysis_for_feedback', analysis_id, result, ttl_hours=24)
        
        return analysis_id
    
    def _update_performance_metrics(self, method: str):
        """Update performance tracking metrics."""
        self.performance_metrics['total_analyses'] += 1
        
        if method not in self.performance_metrics['analyzer_usage']:
            self.performance_metrics['analyzer_usage'][method] = 0
        self.performance_metrics['analyzer_usage'][method] += 1
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty/default result structure."""
        return {
            'predicted_sentiment': 'neutral',
            'confidence': 0.0,
            'sentiment_scores': {'positive': 0.33, 'neutral': 0.34, 'negative': 0.33},
            'method': 'none',
            'error': 'No text provided or analysis failed'
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive performance report for the analyzer.
        
        Returns:
            Performance metrics and statistics
        """
        return {
            'performance_metrics': self.performance_metrics,
            'model_weights': self.model_weights,
            'available_analyzers': list(self.analyzers.keys()),
            'feedback_enabled': self.enable_feedback,
            'auto_retrain_enabled': self.auto_retrain,
            'config': self.config
        }
    
    def update_model_weights(self, new_weights: Dict[str, float]):
        """
        Update model weights for ensemble voting.
        
        Args:
            new_weights: Dictionary of analyzer names to weight values
        """
        # Normalize weights to sum to 1
        total = sum(new_weights.values())
        if total > 0:
            self.model_weights = {k: v/total for k, v in new_weights.items()}
            logger.info(f"Model weights updated: {self.model_weights}")
    
    async def analyze_sentiment_async(self, text: str, method: str = 'auto') -> Dict[str, Any]:
        """
        Async version of sentiment analysis for better performance.
        
        Args:
            text: Text to analyze
            method: Analysis method
            
        Returns:
            Sentiment analysis results
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.analyze_sentiment, text, method)


# Singleton instance for easy access
_unified_analyzer = None


def get_unified_analyzer(reset: bool = False) -> UnifiedSentimentAnalyzer:
    """
    Get or create the unified sentiment analyzer instance.
    
    Args:
        reset: Whether to create a new instance
        
    Returns:
        UnifiedSentimentAnalyzer instance
    """
    global _unified_analyzer
    
    if _unified_analyzer is None or reset:
        _unified_analyzer = UnifiedSentimentAnalyzer()
    
    return _unified_analyzer
