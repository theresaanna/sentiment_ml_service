"""
Feedback Collection System

Collects user corrections when the ML model makes mistakes,
storing them for retraining and continuous improvement.
"""

import os
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import uuid
from app.cache import cache


class FeedbackCollector:
    """Manages user feedback on sentiment predictions"""
    
    def __init__(self, feedback_dir: str = "data/feedback"):
        """
        Initialize the feedback collector
        
        Args:
            feedback_dir: Directory to store feedback data
        """
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        
        # Current feedback file
        today = datetime.now().strftime("%Y%m%d")
        self.feedback_file = self.feedback_dir / f"feedback_{today}.csv"
        
        # Initialize file if it doesn't exist
        self._initialize_feedback_file()
    
    def _initialize_feedback_file(self):
        """Create feedback file with headers if it doesn't exist"""
        if not self.feedback_file.exists():
            headers = [
                'feedback_id',
                'timestamp',
                'video_id',
                'comment_id',
                'comment_text',
                'original_prediction',
                'original_confidence',
                'user_correction',
                'user_confidence',
                'model_version',
                'user_session',
                'additional_notes'
            ]
            
            with open(self.feedback_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
    
    def collect_feedback(self,
                        video_id: str,
                        comment_id: str,
                        comment_text: str,
                        original_prediction: str,
                        original_confidence: float,
                        user_correction: str,
                        user_confidence: Optional[int] = None,
                        model_version: Optional[str] = None,
                        user_session: Optional[str] = None,
                        additional_notes: Optional[str] = None) -> str:
        """
        Collect user feedback on a prediction
        
        Args:
            video_id: YouTube video ID
            comment_id: Unique comment identifier
            comment_text: The comment text
            original_prediction: Model's original sentiment prediction
            original_confidence: Model's confidence score
            user_correction: User's corrected sentiment label
            user_confidence: User's confidence in correction (1-5)
            model_version: Version of the model that made the prediction
            user_session: Session identifier for tracking users
            additional_notes: Any additional notes from the user
            
        Returns:
            Feedback ID for tracking
        """
        feedback_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()
        
        # Prepare feedback record
        feedback_record = {
            'feedback_id': feedback_id,
            'timestamp': timestamp,
            'video_id': video_id,
            'comment_id': comment_id,
            'comment_text': comment_text,
            'original_prediction': original_prediction,
            'original_confidence': original_confidence,
            'user_correction': user_correction,
            'user_confidence': user_confidence or 4,
            'model_version': model_version or 'unknown',
            'user_session': user_session or 'anonymous',
            'additional_notes': additional_notes or ''
        }
        
        # Append to CSV file
        with open(self.feedback_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=feedback_record.keys())
            writer.writerow(feedback_record)
        
        # Also cache the feedback for quick access
        cache_key = f"feedback:{video_id}:{comment_id}"
        cache.set('feedback', cache_key, feedback_record, ttl_hours=24)
        
        # Update statistics
        self._update_feedback_stats()
        
        print(f"Feedback collected: {feedback_id} - {original_prediction} → {user_correction}")
        
        return feedback_id
    
    def _update_feedback_stats(self):
        """Update feedback statistics in cache"""
        stats = cache.get('feedback_stats', 'global') or {
            'total_feedback': 0,
            'corrections_by_type': {},
            'last_updated': None
        }
        
        stats['total_feedback'] += 1
        stats['last_updated'] = datetime.now().isoformat()
        
        cache.set('feedback_stats', 'global', stats, ttl_hours=24)
    
    def get_feedback_summary(self, days: int = 7) -> Dict[str, Any]:
        """
        Get summary of recent feedback
        
        Args:
            days: Number of days to look back
            
        Returns:
            Summary statistics
        """
        all_feedback = []
        
        # Read feedback files from the last N days
        for i in range(days):
            date = datetime.now() - datetime.timedelta(days=i)
            date_str = date.strftime("%Y%m%d")
            feedback_file = self.feedback_dir / f"feedback_{date_str}.csv"
            
            if feedback_file.exists():
                with open(feedback_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    all_feedback.extend(list(reader))
        
        if not all_feedback:
            return {
                'total_feedback': 0,
                'period_days': days,
                'message': 'No feedback collected yet'
            }
        
        # Calculate statistics
        total = len(all_feedback)
        
        # Correction patterns
        correction_matrix = {}
        for fb in all_feedback:
            orig = fb['original_prediction']
            corr = fb['user_correction']
            key = f"{orig}_to_{corr}"
            correction_matrix[key] = correction_matrix.get(key, 0) + 1
        
        # Confidence distribution
        confidence_dist = {}
        for fb in all_feedback:
            conf = fb.get('user_confidence', 'unknown')
            confidence_dist[conf] = confidence_dist.get(conf, 0) + 1
        
        return {
            'total_feedback': total,
            'period_days': days,
            'daily_average': total / days,
            'correction_patterns': correction_matrix,
            'confidence_distribution': confidence_dist,
            'most_common_correction': max(correction_matrix.items(), key=lambda x: x[1])[0] if correction_matrix else None
        }
    
    def export_for_training(self, output_file: Optional[str] = None) -> str:
        """
        Export feedback data in format suitable for model training
        
        Args:
            output_file: Optional output file path
            
        Returns:
            Path to exported file
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.feedback_dir / f"training_from_feedback_{timestamp}.csv"
        
        output_file = Path(output_file)
        
        # Collect all feedback
        all_feedback = []
        for feedback_file in self.feedback_dir.glob("feedback_*.csv"):
            with open(feedback_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                all_feedback.extend(list(reader))
        
        # Convert to training format
        training_data = []
        for fb in all_feedback:
            training_record = {
                'comment_id': fb['comment_id'],
                'comment_text': fb['comment_text'],
                'cleaned_text': fb['comment_text'],  # Could add cleaning here
                'sentiment_label': fb['user_correction'],
                'confidence': fb.get('user_confidence', 4),
                'source': 'user_feedback',
                'original_prediction': fb['original_prediction'],
                'feedback_id': fb['feedback_id'],
                'timestamp': fb['timestamp']
            }
            training_data.append(training_record)
        
        # Write training file
        if training_data:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=training_data[0].keys())
                writer.writeheader()
                writer.writerows(training_data)
            
            print(f"Exported {len(training_data)} feedback records to: {output_file}")
        else:
            print("No feedback data to export")
        
        return str(output_file)
    
    def get_problematic_patterns(self, min_occurrences: int = 3) -> List[Dict[str, Any]]:
        """
        Identify patterns where the model frequently makes mistakes
        
        Args:
            min_occurrences: Minimum number of times a pattern must occur
            
        Returns:
            List of problematic patterns
        """
        # Collect all feedback
        all_feedback = []
        for feedback_file in self.feedback_dir.glob("feedback_*.csv"):
            with open(feedback_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                all_feedback.extend(list(reader))
        
        # Analyze patterns
        error_patterns = {}
        
        for fb in all_feedback:
            # Pattern 1: Specific prediction errors
            error_key = f"{fb['original_prediction']}→{fb['user_correction']}"
            if error_key not in error_patterns:
                error_patterns[error_key] = {
                    'pattern': error_key,
                    'count': 0,
                    'examples': [],
                    'avg_confidence': 0
                }
            
            error_patterns[error_key]['count'] += 1
            error_patterns[error_key]['examples'].append(fb['comment_text'][:100])
            
            # Keep only first 3 examples
            if len(error_patterns[error_key]['examples']) > 3:
                error_patterns[error_key]['examples'] = error_patterns[error_key]['examples'][:3]
        
        # Filter by minimum occurrences
        problematic = [
            pattern for pattern in error_patterns.values()
            if pattern['count'] >= min_occurrences
        ]
        
        # Sort by frequency
        problematic.sort(key=lambda x: x['count'], reverse=True)
        
        return problematic


class FeedbackManager:
    """High-level manager for feedback collection and model improvement"""
    
    def __init__(self):
        self.collector = FeedbackCollector()
        self.training_threshold = 50  # Minimum feedback for retraining
    
    def should_retrain(self) -> bool:
        """
        Check if we have enough feedback to warrant retraining
        
        Returns:
            True if retraining is recommended
        """
        summary = self.collector.get_feedback_summary(days=30)
        return summary['total_feedback'] >= self.training_threshold
    
    def prepare_retraining_data(self, 
                               original_training_files: List[str],
                               include_feedback: bool = True) -> List[str]:
        """
        Prepare combined dataset for retraining
        
        Args:
            original_training_files: Original annotated training files
            include_feedback: Whether to include user feedback
            
        Returns:
            List of training file paths
        """
        training_files = original_training_files.copy()
        
        if include_feedback:
            # Export feedback to training format
            feedback_file = self.collector.export_for_training()
            if feedback_file and Path(feedback_file).exists():
                training_files.append(feedback_file)
                print(f"Added feedback data: {feedback_file}")
        
        return training_files
    
    def get_improvement_metrics(self) -> Dict[str, Any]:
        """
        Calculate metrics showing model improvement from feedback
        
        Returns:
            Dictionary of improvement metrics
        """
        summary = self.collector.get_feedback_summary(days=30)
        patterns = self.collector.get_problematic_patterns()
        
        metrics = {
            'total_corrections': summary['total_feedback'],
            'daily_average': summary.get('daily_average', 0),
            'problematic_patterns': len(patterns),
            'top_issues': patterns[:3] if patterns else [],
            'ready_for_retraining': self.should_retrain()
        }
        
        # Calculate accuracy improvement potential
        if summary['total_feedback'] > 0:
            # This is a simplified metric - in practice you'd compare
            # model performance before and after retraining
            metrics['potential_improvement'] = min(
                summary['total_feedback'] / 1000 * 100,  # Up to 10% per 100 corrections
                15.0  # Cap at 15% improvement
            )
        
        return metrics


# Singleton instance for easy access
feedback_manager = FeedbackManager()
