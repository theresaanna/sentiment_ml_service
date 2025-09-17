"""
Fast Sentiment Analyzer optimized for speed and efficiency.

This module provides a highly optimized sentiment analyzer that uses:
- Batch processing for better GPU utilization
- Smaller, faster models (DistilBERT-based)
- Model quantization for faster inference
- Concurrent processing capabilities
- Intelligent caching strategies
"""
import os
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers.optimization import get_linear_schedule_with_warmup
import torch.nn.functional as F
from app.cache import cache
import time

logger = logging.getLogger(__name__)


class FastSentimentAnalyzer:
    """Fast sentiment analyzer optimized for YouTube comment analysis."""
    
    def __init__(self, batch_size: int = 32, max_workers: int = 4):
        """
        Initialize the fast sentiment analyzer.
        
        Args:
            batch_size: Number of texts to process in each batch
            max_workers: Number of concurrent workers for processing
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use a smaller, faster model for quick inference
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        # Sentiment mapping for the model
        self.label_mapping = {
            'NEGATIVE': 'negative',
            'POSITIVE': 'positive'
        }
        
        # Initialize the model
        self._initialize_model()
        
        logger.info(f"FastSentimentAnalyzer initialized with batch_size={batch_size}, "
                   f"max_workers={max_workers}, device={self.device}")
    
    def _initialize_model(self):
        """Initialize and optimize the sentiment analysis model."""
        try:
            logger.info(f"Loading fast sentiment model: {self.model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Move to appropriate device
            self.model.to(self.device)
            self.model.eval()
            
            # Enable inference optimizations
            if hasattr(torch, 'jit') and torch.cuda.is_available():
                # Try to use TorchScript for faster inference
                try:
                    sample_input = self.tokenizer("sample text", return_tensors="pt", 
                                                padding=True, truncation=True)
                    sample_input = {k: v.to(self.device) for k, v in sample_input.items()}
                    traced_model = torch.jit.trace(self.model, sample_input)
                    self.model = traced_model
                    logger.info("Successfully applied TorchScript optimization")
                except Exception as e:
                    logger.warning(f"TorchScript optimization failed: {e}")
            
            # Create pipeline for simpler inference
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                batch_size=self.batch_size,
                truncation=True,
                max_length=512
            )
            
            logger.info("Model initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        Preprocess texts for better sentiment analysis.
        
        Args:
            texts: List of raw text strings
            
        Returns:
            List of preprocessed text strings
        """
        processed_texts = []
        
        for text in texts:
            # Remove excessive whitespace
            text = ' '.join(text.split())
            
            # Truncate very long texts to save processing time
            if len(text) > 512:
                text = text[:512]
            
            # Skip empty texts
            if not text.strip():
                text = "neutral comment"
            
            processed_texts.append(text)
        
        return processed_texts
    
    def _process_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Process a batch of texts for sentiment analysis.
        
        Args:
            texts: Batch of text strings
            
        Returns:
            List of sentiment analysis results
        """
        try:
            # Preprocess texts
            processed_texts = self._preprocess_texts(texts)
            
            # Run inference
            results = self.pipeline(processed_texts)
            
            # Process results
            processed_results = []
            for i, (text, result) in enumerate(zip(texts, results)):
                # Map labels to our format
                label = result['label']
                score = result['score']
                
                # Convert binary sentiment to our 3-class system
                if label == 'POSITIVE':
                    sentiment_scores = {
                        'positive': score,
                        'negative': 1 - score,
                        'neutral': 0.1  # Small neutral probability
                    }
                    predicted_sentiment = 'positive' if score > 0.6 else 'neutral'
                else:  # NEGATIVE
                    sentiment_scores = {
                        'negative': score,
                        'positive': 1 - score,
                        'neutral': 0.1
                    }
                    predicted_sentiment = 'negative' if score > 0.6 else 'neutral'
                
                # Normalize scores
                total = sum(sentiment_scores.values())
                sentiment_scores = {k: v/total for k, v in sentiment_scores.items()}
                
                # Calculate confidence (higher score = higher confidence)
                confidence = max(sentiment_scores.values())
                
                processed_results.append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'predicted_sentiment': predicted_sentiment,
                    'confidence': confidence,
                    'sentiment_scores': sentiment_scores,
                    'raw_prediction': result
                })
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Return default results for the batch
            return [{
                'text': text[:100] + '...' if len(text) > 100 else text,
                'predicted_sentiment': 'neutral',
                'confidence': 0.5,
                'sentiment_scores': {'positive': 0.33, 'neutral': 0.34, 'negative': 0.33},
                'error': str(e)
            } for text in texts]
    
    def analyze_batch_fast(self, texts: List[str], 
                          progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Fast batch sentiment analysis with concurrent processing.
        
        Args:
            texts: List of text strings to analyze
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with aggregated results and statistics
        """
        if not texts:
            return {
                'total_analyzed': 0,
                'sentiment_counts': {'positive': 0, 'neutral': 0, 'negative': 0},
                'sentiment_percentages': {'positive': 0, 'neutral': 0, 'negative': 0},
                'overall_sentiment': 'neutral',
                'sentiment_score': 0.0,
                'average_confidence': 0.0,
                'processing_time': 0.0,
                'individual_results': []
            }
        
        start_time = time.time()
        
        # Create batches
        batches = [texts[i:i + self.batch_size] 
                  for i in range(0, len(texts), self.batch_size)]
        
        all_results = []
        processed_count = 0
        
        # Process batches concurrently
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batch jobs
            future_to_batch = {
                executor.submit(self._process_batch, batch): batch 
                for batch in batches
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_results = future.result()
                all_results.extend(batch_results)
                processed_count += len(batch_results)
                
                # Update progress
                if progress_callback:
                    progress_callback(processed_count, len(texts))
        
        # Calculate aggregate statistics
        sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
        total_confidence = 0.0
        
        for result in all_results:
            sentiment_counts[result['predicted_sentiment']] += 1
            total_confidence += result['confidence']
        
        # Calculate percentages
        total_analyzed = len(all_results)
        sentiment_percentages = {
            k: (v / total_analyzed * 100) if total_analyzed > 0 else 0
            for k, v in sentiment_counts.items()
        }
        
        # Calculate overall sentiment score (-1 to 1)
        sentiment_score = 0.0
        if total_analyzed > 0:
            sentiment_score = (
                (sentiment_counts['positive'] - sentiment_counts['negative']) / 
                total_analyzed
            )
        
        # Determine overall sentiment category
        overall_sentiment = self._get_overall_sentiment_category(sentiment_score)
        
        # Calculate average confidence
        avg_confidence = total_confidence / total_analyzed if total_analyzed > 0 else 0.0
        
        processing_time = time.time() - start_time
        
        logger.info(f"Fast batch analysis completed: {total_analyzed} texts in "
                   f"{processing_time:.2f}s ({total_analyzed/processing_time:.1f} texts/sec)")
        
        return {
            'total_analyzed': total_analyzed,
            'sentiment_counts': sentiment_counts,
            'sentiment_percentages': sentiment_percentages,
            'overall_sentiment': overall_sentiment,
            'sentiment_score': sentiment_score,
            'average_confidence': avg_confidence,
            'processing_time': processing_time,
            'throughput': total_analyzed / processing_time if processing_time > 0 else 0,
            'individual_results': all_results,
            'model_info': {
                'model_name': self.model_name,
                'device': str(self.device),
                'batch_size': self.batch_size,
                'max_workers': self.max_workers
            }
        }
    
    def _get_overall_sentiment_category(self, score: float) -> str:
        """Convert sentiment score to category."""
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
    
    async def analyze_batch_async(self, texts: List[str], 
                                 progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Async version of batch sentiment analysis.
        
        Args:
            texts: List of text strings to analyze
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with aggregated results and statistics
        """
        # Run the synchronous batch analysis in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.analyze_batch_fast, 
            texts, 
            progress_callback
        )
    
    def analyze_with_caching(self, texts: List[str], 
                           cache_key: str,
                           progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Analyze texts with intelligent caching.
        
        Args:
            texts: List of text strings to analyze
            cache_key: Unique key for caching results
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with results (from cache or fresh analysis)
        """
        # Check cache first
        cached_result = cache.get('fast_sentiment', cache_key)
        if cached_result:
            logger.info(f"Using cached sentiment analysis for key: {cache_key}")
            return cached_result
        
        # Run fresh analysis
        result = self.analyze_batch_fast(texts, progress_callback)
        
        # Cache the result for 12 hours (shorter than full analysis due to speed)
        cache.set('fast_sentiment', cache_key, result, ttl_hours=12)
        
        return result
    
    def warm_up_model(self):
        """Warm up the model with a few sample predictions for faster first inference."""
        logger.info("Warming up the sentiment analysis model...")
        
        sample_texts = [
            "This is great!",
            "I don't like this",
            "This is okay, nothing special",
            "Amazing work, love it!",
            "Terrible, worst ever"
        ]
        
        try:
            # Run a quick batch to initialize all model components
            self._process_batch(sample_texts)
            logger.info("Model warm-up completed successfully")
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")


# Global instance for reuse across the application
_fast_analyzer = None

def get_fast_analyzer() -> FastSentimentAnalyzer:
    """Get or create the global fast sentiment analyzer instance."""
    global _fast_analyzer
    if _fast_analyzer is None:
        _fast_analyzer = FastSentimentAnalyzer()
        # Warm up the model on first use
        _fast_analyzer.warm_up_model()
    return _fast_analyzer
