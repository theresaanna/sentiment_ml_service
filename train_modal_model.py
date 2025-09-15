#!/usr/bin/env python3
"""
Train Sentiment Model for Modal Deployment

This script trains a sentiment classification model from annotated Lady Gaga 
comment data and prepares it for deployment on Modal.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from app_modules.ml.model_trainer import train_from_csv, SentimentModelTrainer

def main():
    """Main training function"""
    
    # Path to the Lady Gaga dataset
    lady_gaga_csv = "/Users/theresa/PycharmProjects/sentiment_analyzer/data/training_data_habpdmFSTOo_Lady Gaga on Double Standards _20250909_190009.csv"
    
    # Check if file exists
    if not Path(lady_gaga_csv).exists():
        print(f"❌ Training data not found: {lady_gaga_csv}")
        return 1
    
    print("🎯 Training Sentiment Model for Modal Deployment")
    print("=" * 60)
    
    # Create models directory if it doesn't exist
    models_dir = Path(__file__).parent / "app_modules" / "ml" / "models"
    models_dir.mkdir(exist_ok=True, parents=True)
    
    # Train the model
    print(f"\n📊 Loading data from: {Path(lady_gaga_csv).name}")
    
    try:
        # Initialize trainer
        trainer = SentimentModelTrainer(model_dir=str(models_dir))
        
        # Load and check data
        df, skipped = trainer.load_training_data([lady_gaga_csv])
        
        print(f"\n📈 Dataset Statistics:")
        print(f"  Total annotated: {len(df)}")
        print(f"  Skipped (unannotated): {skipped}")
        
        # Check sentiment distribution
        sentiment_counts = df['sentiment_label'].value_counts()
        print(f"\n📊 Sentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            print(f"  {sentiment}: {count} ({count/len(df)*100:.1f}%)")
        
        if len(df) < 50:
            print(f"\n⚠️  Warning: Only {len(df)} training samples. Consider annotating more data.")
            print("   Run the interactive annotator to add more annotations:")
            print("   cd ../sentiment_analyzer")
            print(f"   python scripts/interactive_annotator.py data/{Path(lady_gaga_csv).name}")
        
        # Prepare features
        print("\n🔧 Preparing features...")
        training_data = trainer.prepare_features(df)
        
        # Train model with different algorithms
        algorithms = ['logistic_regression', 'random_forest', 'naive_bayes']
        best_algorithm = None
        best_accuracy = 0
        
        print("\n🚀 Training models...")
        for algorithm in algorithms:
            print(f"\n--- Training {algorithm} ---")
            metrics = trainer.train_model(training_data, algorithm=algorithm)
            
            if metrics['test_accuracy'] > best_accuracy:
                best_accuracy = metrics['test_accuracy']
                best_algorithm = algorithm
        
        # Train final model with best algorithm
        print(f"\n🏆 Best algorithm: {best_algorithm} (accuracy: {best_accuracy:.3f})")
        print("Training final model...")
        
        trainer.train_model(training_data, algorithm=best_algorithm)
        
        # Save the model
        model_path = trainer.save_model(model_name="modal_sentiment_model")
        
        print(f"\n✅ Model training complete!")
        print(f"   Model saved to: {model_path}")
        print(f"   Algorithm: {best_algorithm}")
        print(f"   Test accuracy: {best_accuracy:.3f}")
        
        # Test the model
        print("\n🧪 Testing model with sample texts...")
        test_model(trainer)
        
        print("\n📦 Next Steps for Modal Deployment:")
        print("1. Deploy to Modal:")
        print("   modal deploy modal_app.py")
        print("\n2. Test the deployment:")
        print("   python test_modal_integration.py")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def test_model(trainer: SentimentModelTrainer):
    """Test the trained model with sample texts"""
    
    test_texts = [
        "Lady Gaga is absolutely amazing! I love her so much!",
        "This is terrible and I hate it",
        "The video was posted in 2020",
        "What a waste of time, absolutely horrible",
        "Great interview, she makes excellent points",
        "I don't understand what she's saying",
        "BEST ARTIST EVER!!!",
        "meh, not impressed",
        "She's so talented and inspiring",
        "This is stupid and boring"
    ]
    
    results = trainer.predict(test_texts)
    
    print("\n📝 Sample predictions:")
    print("-" * 60)
    
    for result in results:
        text = result['text'][:50]
        sentiment = result['sentiment']
        confidence = result.get('confidence', 0)
        
        # Emoji for sentiment
        emoji = {'positive': '😊', 'negative': '😞', 'neutral': '😐'}.get(sentiment, '❓')
        
        print(f"{emoji} {sentiment:8s} (conf: {confidence:.2f}) | {text}")
    
    print("-" * 60)


if __name__ == "__main__":
    sys.exit(main())