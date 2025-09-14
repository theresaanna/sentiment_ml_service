"""
Unit tests for sentiment analyzer classes.
"""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import torch
from app.science.sentiment_analyzer import SentimentAnalyzer
from app.ml.ml_sentiment_analyzer import MLSentimentAnalyzer


class TestSentimentAnalyzer:
    """Test the main SentimentAnalyzer class."""
    
    @patch('app.science.sentiment_analyzer.get_model_manager')
    def test_initialization(self, mock_get_manager):
        """Test SentimentAnalyzer initialization."""
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager
        
        analyzer = SentimentAnalyzer(batch_size=16)
        
        assert analyzer.batch_size == 16
        assert analyzer.roberta_weight == 0.7
        assert analyzer.gb_weight == 0.3
        assert analyzer.sentiment_labels == ['negative', 'neutral', 'positive']
        mock_get_manager.assert_called_once()
    
    @patch('app.science.sentiment_analyzer.get_model_manager')
    @patch('app.science.sentiment_analyzer.cache')
    def test_analyze_sentiment_cached(self, mock_cache, mock_get_manager):
        """Test sentiment analysis with cached result."""
        # Setup mocks
        mock_cache.get.return_value = {
            'label': 'positive',
            'scores': {'negative': 0.1, 'neutral': 0.2, 'positive': 0.7},
            'confidence': 0.7,
            'model': 'roberta'
        }
        
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_sentiment("Great product!")
        
        assert result['label'] == 'positive'
        assert result['confidence'] == 0.7
        mock_cache.get.assert_called_once()
    
    @patch('app.science.sentiment_analyzer.os.path.exists')
    @patch('app.science.sentiment_analyzer.get_model_manager')
    @patch('app.science.sentiment_analyzer.cache')
    def test_analyze_sentiment_uncached(self, mock_cache, mock_get_manager, mock_path_exists):
        """Test sentiment analysis without cache."""
        # Setup mocks
        mock_path_exists.return_value = False  # Prevent GB model from loading
        mock_cache.get.return_value = None
        mock_manager = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        
        # Mock tokenizer output
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        
        # Mock model output - use raw logits that will become [0.1, 0.2, 0.7] after softmax
        mock_output = MagicMock()
        # These logits will produce approximately [0.09, 0.13, 0.78] after softmax
        mock_output.logits = torch.tensor([[-2.0, -1.5, 1.5]])
        mock_model.return_value = mock_output
        mock_model.device = 'cpu'
        
        mock_manager.get_roberta_sentiment.return_value = (mock_tokenizer, mock_model)
        mock_get_manager.return_value = mock_manager
        
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_sentiment("Great product!")
        
        assert result['label'] == 'positive'
        assert result['confidence'] > 0.6
        assert result['model'] == 'roberta'
        mock_cache.set.assert_called_once()
    
    @patch('app.science.sentiment_analyzer.get_model_manager')
    @patch('app.science.sentiment_analyzer.cache')
    def test_analyze_sentiment_with_gb_model(self, mock_cache, mock_get_manager):
        """Test sentiment analysis with gradient boosting ensemble."""
        # Setup mocks
        mock_cache.get.return_value = None
        mock_manager = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        
        # Mock tokenizer and model
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.1, 0.2, 0.7]])
        mock_model.return_value = mock_output
        mock_model.device = 'cpu'
        
        mock_manager.get_roberta_sentiment.return_value = (mock_tokenizer, mock_model)
        mock_get_manager.return_value = mock_manager
        
        analyzer = SentimentAnalyzer()
        
        # Mock GB model
        analyzer.gb_model = MagicMock()
        analyzer.gb_model.predict_proba.return_value = np.array([[0.2, 0.3, 0.5]])
        analyzer.gb_vectorizer = MagicMock()
        analyzer.gb_vectorizer.transform.return_value = MagicMock()
        
        result = analyzer.analyze_sentiment("Great product!")
        
        assert result['label'] in ['positive', 'neutral', 'negative']
        assert result['model'] == 'ensemble'
        assert 'scores' in result
    
    @patch('app.science.sentiment_analyzer.get_model_manager')
    def test_analyze_batch_empty(self, mock_get_manager):
        """Test batch analysis with empty input."""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_batch([])
        
        assert result['total_analyzed'] == 0
        assert result['overall_sentiment'] == 'neutral'
        assert result['average_confidence'] == 0.0
        assert len(result['individual_results']) == 0
    
    @patch('app.science.sentiment_analyzer.get_model_manager')
    @patch('app.science.sentiment_analyzer.cache')
    def test_analyze_batch_with_texts(self, mock_cache, mock_get_manager):
        """Test batch analysis with multiple texts."""
        # Setup mocks
        mock_cache.get.return_value = None
        mock_manager = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        
        # Mock tokenizer and model
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.1, 0.2, 0.7]])
        mock_model.return_value = mock_output
        mock_model.device = 'cpu'
        
        mock_manager.get_roberta_sentiment.return_value = (mock_tokenizer, mock_model)
        mock_get_manager.return_value = mock_manager
        
        analyzer = SentimentAnalyzer()
        texts = ["Great!", "Terrible!", "Okay"]
        result = analyzer.analyze_batch(texts)
        
        assert result['total_analyzed'] == 3
        assert 'overall_sentiment' in result
        assert 'distribution' in result
        assert len(result['individual_results']) == 3
    
    @patch('app.science.sentiment_analyzer.get_model_manager')
    def test_analyze_batch_with_progress_callback(self, mock_get_manager):
        """Test batch analysis with progress callback."""
        mock_manager = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        
        # Mock tokenizer and model
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.1, 0.2, 0.7]])
        mock_model.return_value = mock_output
        mock_model.device = 'cpu'
        
        mock_manager.get_roberta_sentiment.return_value = (mock_tokenizer, mock_model)
        mock_get_manager.return_value = mock_manager
        
        analyzer = SentimentAnalyzer(batch_size=2)
        
        progress_calls = []
        def progress_callback(current, total):
            progress_calls.append((current, total))
        
        texts = ["Text1", "Text2", "Text3", "Text4"]
        result = analyzer.analyze_batch(texts, progress_callback=progress_callback)
        
        assert len(progress_calls) > 0
        assert result['total_analyzed'] == 4


class TestMLSentimentAnalyzer:
    """Test the MLSentimentAnalyzer class."""
    
    @patch('app.ml.ml_sentiment_analyzer.Path')
    def test_initialization_no_model(self, mock_path):
        """Test MLSentimentAnalyzer initialization without model."""
        mock_path.return_value.exists.return_value = False
        
        analyzer = MLSentimentAnalyzer(use_fallback=False)
        
        assert analyzer.model is None
        assert analyzer.predictions_made == 0
        assert analyzer.feedback_received == 0
    
    @patch('app.ml.ml_sentiment_analyzer.joblib')
    @patch('app.ml.ml_sentiment_analyzer.Path')
    def test_initialization_with_model(self, mock_path, mock_joblib):
        """Test MLSentimentAnalyzer initialization with model."""
        mock_path.return_value.exists.return_value = True
        mock_model = MagicMock()
        mock_joblib.load.return_value = mock_model
        
        analyzer = MLSentimentAnalyzer(model_path="test_model.pkl", use_fallback=False)
        
        assert analyzer.model is not None
        mock_joblib.load.assert_called_once()
    
    @patch('app.ml.ml_sentiment_analyzer.SentimentAnalyzer')
    def test_initialization_with_fallback(self, mock_sentiment_analyzer):
        """Test initialization with fallback analyzer."""
        analyzer = MLSentimentAnalyzer(use_fallback=True)
        
        assert analyzer.fallback_analyzer is not None
        mock_sentiment_analyzer.assert_called_once()
    
    def test_analyze_sentiment_empty_text(self):
        """Test sentiment analysis with empty text."""
        analyzer = MLSentimentAnalyzer(use_fallback=False)
        
        result = analyzer.analyze_sentiment("")
        assert result['sentiment'] == 'neutral'
        assert result['confidence'] == 0.0
        assert result['method'] == 'empty_text'
        
        result = analyzer.analyze_sentiment("   ")
        assert result['sentiment'] == 'neutral'
        assert result['confidence'] == 0.0
    
    @patch('app.ml.ml_sentiment_analyzer.joblib')
    @patch('app.ml.ml_sentiment_analyzer.Path')
    def test_analyze_sentiment_with_model(self, mock_path, mock_joblib):
        """Test sentiment analysis with ML model."""
        mock_path.return_value.exists.return_value = True
        
        # Create mock model with pipeline structure
        mock_classifier = MagicMock()
        mock_classifier.predict_proba.return_value = [[0.1, 0.2, 0.7]]
        
        mock_model = MagicMock()
        mock_model.predict.return_value = [1]  # positive
        mock_model.named_steps = {'classifier': mock_classifier}
        
        mock_joblib.load.return_value = mock_model
        
        analyzer = MLSentimentAnalyzer(use_fallback=False)
        result = analyzer.analyze_sentiment("Great product!")
        
        assert result['sentiment'] == 'positive'
        assert result['method'] == 'ml_model'
        assert 'confidence' in result
        assert 'probabilities' in result
        assert analyzer.predictions_made == 1
    
    @patch('app.ml.ml_sentiment_analyzer.SentimentAnalyzer')
    def test_analyze_sentiment_with_fallback(self, mock_sentiment_analyzer):
        """Test sentiment analysis falling back to rule-based analyzer."""
        mock_fallback = MagicMock()
        mock_fallback.analyze_sentiment.return_value = {
            'overall_sentiment': 'positive',
            'confidence': 0.8,
            'subjectivity': 0.6,
            'emotions': {'joy': 0.7}
        }
        mock_sentiment_analyzer.return_value = mock_fallback
        
        analyzer = MLSentimentAnalyzer(use_fallback=True)
        analyzer.model = None  # Force fallback
        
        result = analyzer.analyze_sentiment("Great product!")
        
        assert result['sentiment'] == 'positive'
        assert result['confidence'] == 0.8
        assert result['method'] == 'rule_based_fallback'
        assert result['subjectivity'] == 0.6
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        analyzer = MLSentimentAnalyzer(use_fallback=False)
        
        # Test URL removal
        text = "Check this out http://example.com and www.test.com"
        cleaned = analyzer._clean_text(text)
        assert "http://" not in cleaned
        assert "www." not in cleaned
        
        # Test whitespace normalization
        text = "Too    many     spaces\n\nand\tnewlines"
        cleaned = analyzer._clean_text(text)
        assert "  " not in cleaned
        assert "\n" not in cleaned
        assert "\t" not in cleaned
        
        # Test strip
        text = "  leading and trailing spaces  "
        cleaned = analyzer._clean_text(text)
        assert cleaned == "leading and trailing spaces"
    
    @patch('app.ml.ml_sentiment_analyzer.joblib')
    @patch('app.ml.ml_sentiment_analyzer.Path')
    def test_extract_features(self, mock_path, mock_joblib):
        """Test feature extraction."""
        mock_path.return_value.exists.return_value = True
        mock_model = MagicMock()
        mock_joblib.load.return_value = mock_model
        
        analyzer = MLSentimentAnalyzer(use_fallback=False)
        
        # Test with various texts
        features = analyzer._extract_features("This is AMAZING!!!")
        assert 'text_length' in features
        assert 'exclamation_marks' in features
        assert 'question_marks' in features
        assert 'capital_ratio' in features
        
        features = analyzer._extract_features("Is this good?")
        assert features['question_marks'] == 1
        
        features = analyzer._extract_features("SHOUTING")
        assert features['capital_ratio'] > 0.9
    
    @patch('app.ml.ml_sentiment_analyzer.joblib')
    @patch('app.ml.ml_sentiment_analyzer.Path')
    def test_ml_predict_error_handling(self, mock_path, mock_joblib):
        """Test error handling in ML prediction."""
        mock_path.return_value.exists.return_value = True
        mock_model = MagicMock()
        mock_model.predict.side_effect = Exception("Model error")
        mock_joblib.load.return_value = mock_model
        
        analyzer = MLSentimentAnalyzer(use_fallback=False)
        result = analyzer.analyze_sentiment("Test text")
        
        assert result['sentiment'] == 'neutral'
        assert result['confidence'] == 0.0
        assert result['method'] == 'default'
    
    def test_load_model_error_handling(self):
        """Test model loading error handling."""
        analyzer = MLSentimentAnalyzer(model_path="nonexistent_model.pkl", use_fallback=False)
        
        # Should handle missing model gracefully
        assert analyzer.model is None
        
        # Test with corrupted model file
        with patch('app.ml.ml_sentiment_analyzer.joblib.load') as mock_load:
            mock_load.side_effect = Exception("Corrupted file")
            result = analyzer.load_model("corrupted.pkl")
            assert result is False
            assert analyzer.model is None


class TestBatchProcessing:
    """Test batch processing functionality."""
    
    @patch('app.science.sentiment_analyzer.get_model_manager')
    def test_batch_sentiment_distribution(self, mock_get_manager):
        """Test sentiment distribution calculation in batch processing."""
        mock_manager = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        
        # Mock different sentiments for each text
        sentiments = [
            torch.tensor([[0.7, 0.2, 0.1]]),  # negative
            torch.tensor([[0.1, 0.7, 0.2]]),  # neutral
            torch.tensor([[0.1, 0.2, 0.7]]),  # positive
            torch.tensor([[0.1, 0.1, 0.8]]),  # positive
        ]
        
        call_count = [0]
        def get_logits(*args, **kwargs):
            output = MagicMock()
            output.logits = sentiments[call_count[0] % len(sentiments)]
            call_count[0] += 1
            return output
        
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        mock_model.side_effect = get_logits
        mock_model.device = 'cpu'
        
        mock_manager.get_roberta_sentiment.return_value = (mock_tokenizer, mock_model)
        mock_get_manager.return_value = mock_manager
        
        analyzer = SentimentAnalyzer()
        texts = ["negative", "neutral", "positive", "very positive"]
        result = analyzer.analyze_batch(texts, use_cache=False)
        
        assert result['total_analyzed'] == 4
        assert 'distribution' in result
        assert sum(result['distribution'].values()) == 4
        assert 'distribution_percentage' in result
        
    @patch('app.science.sentiment_analyzer.get_model_manager')
    def test_batch_confidence_calculation(self, mock_get_manager):
        """Test average confidence calculation in batch processing."""
        mock_manager = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        mock_output = MagicMock()
        # Use raw logits that will produce high confidence after softmax
        mock_output.logits = torch.tensor([[-3.0, -3.0, 2.0]])  # Will give ~[0.006, 0.006, 0.988] after softmax
        mock_model.return_value = mock_output
        mock_model.device = 'cpu'
        
        mock_manager.get_roberta_sentiment.return_value = (mock_tokenizer, mock_model)
        mock_get_manager.return_value = mock_manager
        
        analyzer = SentimentAnalyzer()
        texts = ["text1", "text2", "text3"]
        result = analyzer.analyze_batch(texts, use_cache=False)
        
        assert 'average_confidence' in result
        assert result['average_confidence'] > 0.7  # High confidence expected