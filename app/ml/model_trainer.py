"""
Machine Learning Model Trainer for Sentiment Analysis

This module handles training sentiment classification models from annotated data.
Uses scikit-learn for the ML pipeline and supports various algorithms.
"""

import os
import pickle
import json
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
import joblib


class SentimentModelTrainer:
    """Train and evaluate sentiment classification models"""
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize the trainer
        
        Args:
            model_dir: Directory to save trained models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.vectorizer = None
        self.classifier = None
        self.pipeline = None
        self.feature_names = None
        self.label_encoder = {'positive': 1, 'neutral': 0, 'negative': -1}
        self.label_decoder = {v: k for k, v in self.label_encoder.items()}
        
        # Training metadata
        self.training_metadata = {}
        
    def load_training_data(self, csv_files: List[str]) -> Tuple[pd.DataFrame, int]:
        """
        Load and combine training data from CSV files
        
        Args:
            csv_files: List of CSV file paths
            
        Returns:
            DataFrame with training data and count of skipped rows
        """
        all_data = []
        skipped = 0
        
        for csv_file in csv_files:
            print(f"Loading data from: {csv_file}")
            
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Only include rows with sentiment labels
                    if row.get('sentiment_label') and row['sentiment_label'].strip():
                        all_data.append(row)
                    else:
                        skipped += 1
        
        df = pd.DataFrame(all_data)
        print(f"Loaded {len(df)} annotated comments, skipped {skipped} unannotated")
        
        # Convert sentiment labels to lowercase for consistency
        df['sentiment_label'] = df['sentiment_label'].str.lower().str.strip()
        
        # Add computed features
        df = self._add_computed_features(df)
        
        return df, skipped
    
    def _add_computed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add computed features to the dataframe"""
        # Convert string numbers to actual numbers
        numeric_columns = ['likes', 'char_count', 'word_count', 'sentence_count', 
                          'exclamation_count', 'question_count', 'emoji_count']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        if 'caps_ratio' in df.columns:
            df['caps_ratio'] = pd.to_numeric(df['caps_ratio'], errors='coerce').fillna(0)
        
        # Add time-based features
        if 'published_at' in df.columns:
            df['published_datetime'] = pd.to_datetime(df['published_at'], errors='coerce')
            df['hour'] = df['published_datetime'].dt.hour
            df['day_of_week'] = df['published_datetime'].dt.dayofweek
        
        # Add engagement features
        if 'likes' in df.columns:
            df['has_likes'] = (df['likes'] > 0).astype(int)
            df['high_engagement'] = (df['likes'] > df['likes'].median()).astype(int)
        
        # Text complexity features
        if 'word_count' in df.columns and 'sentence_count' in df.columns:
            df['avg_words_per_sentence'] = df['word_count'] / df['sentence_count'].replace(0, 1)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, text_column: str = 'cleaned_text') -> Dict[str, Any]:
        """
        Prepare features for training
        
        Args:
            df: DataFrame with training data
            text_column: Column containing text to analyze
            
        Returns:
            Dictionary with X (features), y (labels), and feature metadata
        """
        # Get text data
        texts = df[text_column].fillna('').tolist()
        
        # Get labels
        labels = df['sentiment_label'].map(self.label_encoder).tolist()
        
        # Get additional features if available
        feature_columns = ['word_count', 'exclamation_count', 'question_count', 
                          'caps_ratio', 'emoji_count', 'has_likes', 'high_engagement']
        
        available_features = [col for col in feature_columns if col in df.columns]
        
        additional_features = None
        if available_features:
            additional_features = df[available_features].fillna(0).values
        
        return {
            'texts': texts,
            'labels': labels,
            'additional_features': additional_features,
            'feature_columns': available_features
        }
    
    def train_model(self, 
                   training_data: Dict[str, Any],
                   algorithm: str = 'logistic_regression',
                   test_size: float = 0.2,
                   random_state: int = 42) -> Dict[str, Any]:
        """
        Train a sentiment classification model
        
        Args:
            training_data: Dictionary with texts, labels, and optional features
            algorithm: ML algorithm to use
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with training results and metrics
        """
        print(f"\nTraining {algorithm} model...")
        
        # Split data
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            training_data['texts'],
            training_data['labels'],
            test_size=test_size,
            random_state=random_state,
            stratify=training_data['labels']
        )
        
        # Create vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # Use unigrams and bigrams
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            use_idf=True
        )
        
        # Select classifier
        if algorithm == 'logistic_regression':
            self.classifier = LogisticRegression(
                C=1.0,
                class_weight='balanced',
                max_iter=1000,
                random_state=random_state
            )
        elif algorithm == 'random_forest':
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=random_state
            )
        elif algorithm == 'svm':
            self.classifier = LinearSVC(
                C=1.0,
                class_weight='balanced',
                random_state=random_state,
                max_iter=2000
            )
        elif algorithm == 'naive_bayes':
            self.classifier = MultinomialNB(alpha=0.1)
        elif algorithm == 'gradient_boosting':
            self.classifier = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.classifier)
        ])
        
        # Train model
        print("Fitting model...")
        self.pipeline.fit(X_train_text, y_train)
        
        # Make predictions
        y_pred_train = self.pipeline.predict(X_train_text)
        y_pred_test = self.pipeline.predict(X_test_text)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        print(f"\nTraining Accuracy: {train_accuracy:.3f}")
        print(f"Test Accuracy: {test_accuracy:.3f}")
        
        # Detailed classification report
        print("\nClassification Report (Test Set):")
        report = classification_report(
            y_test, y_pred_test,
            target_names=['negative', 'neutral', 'positive'],
            output_dict=True
        )
        
        # Print formatted report
        for label in ['negative', 'neutral', 'positive']:
            if label in report:
                metrics = report[label]
                print(f"  {label:10s}: precision={metrics['precision']:.3f}, "
                      f"recall={metrics['recall']:.3f}, f1={metrics['f1-score']:.3f}")
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred_test)
        
        # Cross-validation
        print("\nCross-validation scores...")
        cv_scores = cross_val_score(
            self.pipeline, 
            training_data['texts'], 
            training_data['labels'],
            cv=5,
            scoring='accuracy'
        )
        print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Store training metadata
        self.training_metadata = {
            'algorithm': algorithm,
            'training_date': datetime.now().isoformat(),
            'training_samples': len(X_train_text),
            'test_samples': len(X_test_text),
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'cv_mean_accuracy': float(cv_scores.mean()),
            'cv_std_accuracy': float(cv_scores.std()),
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'feature_columns': training_data.get('feature_columns', []),
            'vectorizer_features': self.vectorizer.get_feature_names_out().tolist()[:100]  # Sample
        }
        
        return self.training_metadata
    
    def hyperparameter_tuning(self, 
                            training_data: Dict[str, Any],
                            algorithm: str = 'logistic_regression') -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            training_data: Dictionary with texts and labels
            algorithm: ML algorithm to tune
            
        Returns:
            Best parameters and scores
        """
        print(f"\nPerforming hyperparameter tuning for {algorithm}...")
        
        # Define parameter grids
        param_grids = {
            'logistic_regression': {
                'classifier__C': [0.01, 0.1, 1.0, 10.0],
                'classifier__penalty': ['l2'],
                'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
                'vectorizer__max_features': [1000, 3000, 5000]
            },
            'random_forest': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5, 10],
                'vectorizer__ngram_range': [(1, 1), (1, 2)]
            },
            'svm': {
                'classifier__C': [0.01, 0.1, 1.0, 10.0],
                'vectorizer__ngram_range': [(1, 1), (1, 2)],
                'vectorizer__max_features': [1000, 3000, 5000]
            }
        }
        
        if algorithm not in param_grids:
            print(f"No parameter grid defined for {algorithm}")
            return {}
        
        # Create pipeline
        if algorithm == 'logistic_regression':
            classifier = LogisticRegression(class_weight='balanced', max_iter=1000)
        elif algorithm == 'random_forest':
            classifier = RandomForestClassifier(class_weight='balanced')
        elif algorithm == 'svm':
            classifier = LinearSVC(class_weight='balanced', max_iter=2000)
        else:
            return {}
        
        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer()),
            ('classifier', classifier)
        ])
        
        # Perform grid search
        grid_search = GridSearchCV(
            pipeline,
            param_grids[algorithm],
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(training_data['texts'], training_data['labels'])
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.3f}")
        
        # Update pipeline with best model
        self.pipeline = grid_search.best_estimator_
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
    
    def save_model(self, model_name: str = None) -> str:
        """
        Save the trained model and metadata
        
        Args:
            model_name: Optional custom name for the model
            
        Returns:
            Path to saved model
        """
        if self.pipeline is None:
            raise ValueError("No model to save. Train a model first.")
        
        # Generate model name if not provided
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            algorithm = self.training_metadata.get('algorithm', 'unknown')
            model_name = f"sentiment_model_{algorithm}_{timestamp}"
        
        # Save model
        model_path = self.model_dir / f"{model_name}.pkl"
        joblib.dump(self.pipeline, model_path)
        print(f"Model saved to: {model_path}")
        
        # Save metadata
        metadata_path = self.model_dir / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.training_metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_path}")
        
        # Save as 'latest' model for easy access
        latest_model_path = self.model_dir / "latest_model.pkl"
        joblib.dump(self.pipeline, latest_model_path)
        
        latest_metadata_path = self.model_dir / "latest_model_metadata.json"
        with open(latest_metadata_path, 'w') as f:
            json.dump(self.training_metadata, f, indent=2)
        
        return str(model_path)
    
    def load_model(self, model_path: str) -> Pipeline:
        """
        Load a saved model
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded pipeline
        """
        self.pipeline = joblib.load(model_path)
        
        # Load metadata if available
        metadata_path = Path(model_path).with_suffix('') + '_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.training_metadata = json.load(f)
        
        return self.pipeline
    
    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Make predictions on new texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of predictions with labels and probabilities
        """
        if self.pipeline is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        # Make predictions
        predictions = self.pipeline.predict(texts)
        
        # Get probabilities if available
        probabilities = None
        if hasattr(self.pipeline.named_steps['classifier'], 'predict_proba'):
            probabilities = self.pipeline.named_steps['classifier'].predict_proba(texts)
        
        # Format results
        results = []
        for i, (text, pred) in enumerate(zip(texts, predictions)):
            result = {
                'text': text,
                'sentiment': self.label_decoder.get(pred, 'neutral'),
                'sentiment_score': float(pred)
            }
            
            if probabilities is not None:
                # Add probability distribution
                prob_dist = probabilities[i]
                result['probabilities'] = {
                    'negative': float(prob_dist[0]) if len(prob_dist) > 0 else 0,
                    'neutral': float(prob_dist[1]) if len(prob_dist) > 1 else 0,
                    'positive': float(prob_dist[2]) if len(prob_dist) > 2 else 0
                }
                result['confidence'] = float(max(prob_dist))
            
            results.append(result)
        
        return results


def train_from_csv(csv_files: List[str], 
                  algorithm: str = 'logistic_regression',
                  output_dir: str = 'models',
                  tune_hyperparameters: bool = False) -> str:
    """
    Convenience function to train a model from CSV files
    
    Args:
        csv_files: List of CSV file paths with annotated data
        algorithm: ML algorithm to use
        output_dir: Directory to save the model
        tune_hyperparameters: Whether to perform hyperparameter tuning
        
    Returns:
        Path to saved model
    """
    # Initialize trainer
    trainer = SentimentModelTrainer(model_dir=output_dir)
    
    # Load data
    df, skipped = trainer.load_training_data(csv_files)
    
    if len(df) < 50:
        print(f"Warning: Only {len(df)} training samples. Consider annotating more data.")
    
    # Prepare features
    training_data = trainer.prepare_features(df)
    
    # Train or tune
    if tune_hyperparameters and len(df) >= 100:
        trainer.hyperparameter_tuning(training_data, algorithm)
    else:
        trainer.train_model(training_data, algorithm)
    
    # Save model
    model_path = trainer.save_model()
    
    print(f"\n✅ Model training complete!")
    print(f"   Model saved to: {model_path}")
    print(f"   Training samples: {len(df)}")
    print(f"   Test accuracy: {trainer.training_metadata.get('test_accuracy', 0):.3f}")
    
    return model_path


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python model_trainer.py <csv_file> [algorithm]")
        print("Algorithms: logistic_regression, random_forest, svm, naive_bayes, gradient_boosting")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    algorithm = sys.argv[2] if len(sys.argv) > 2 else 'logistic_regression'
    
    model_path = train_from_csv([csv_file], algorithm=algorithm)
    print(f"Model saved to: {model_path}")
