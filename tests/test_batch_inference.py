"""
Test script for batch inference optimization.
"""
import time
import random
import string
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml.batch_processor import BatchInferenceOptimizer, BatchConfig
from app.ml.ml_sentiment_analyzer import MLSentimentAnalyzer
from app.science.fast_sentiment_analyzer import FastSentimentAnalyzer


def generate_test_comments(count: int = 1000) -> list:
    """Generate test comments for benchmarking."""
    sentiments = [
        "This video is amazing! I love it!",
        "Terrible content, waste of time.",
        "Not bad, but could be better.",
        "Absolutely fantastic work!",
        "I disagree with this completely.",
        "Neutral comment about the video.",
        "Best video I've ever seen!",
        "Worst video ever made.",
        "It's okay, nothing special.",
        "Great tutorial, very helpful!",
        "This doesn't make any sense.",
        "I learned so much from this.",
        "Boring and uninformative.",
        "Perfect explanation, thank you!",
        "Could use more examples.",
        "Outstanding quality content!",
        "Not worth watching.",
        "Interesting perspective.",
        "This changed my mind completely!",
        "Same old stuff, nothing new."
    ]
    
    comments = []
    for _ in range(count):
        # Random variation of base sentiments
        base = random.choice(sentiments)
        variation = ''.join(random.choices(string.ascii_lowercase + ' ', k=random.randint(0, 20)))
        comments.append(f"{base} {variation}".strip())
    
    return comments


def benchmark_batch_sizes():
    """Benchmark different batch sizes."""
    print("=" * 60)
    print("Benchmarking Batch Sizes")
    print("=" * 60)
    
    # Generate test data
    test_comments = generate_test_comments(1000)
    
    # Initialize analyzer
    try:
        analyzer = MLSentimentAnalyzer()
    except Exception as e:
        print(f"Warning: Could not initialize MLSentimentAnalyzer: {e}")
        print("Using mock analyzer for testing")
        
        class MockAnalyzer:
            def analyze_sentiment(self, text):
                return {
                    'sentiment': random.choice(['positive', 'negative', 'neutral']),
                    'confidence': random.random(),
                    'method': 'mock'
                }
            
            def analyze_batch(self, texts):
                return [self.analyze_sentiment(text) for text in texts]
        
        analyzer = MockAnalyzer()
    
    # Test different batch sizes
    batch_sizes = [8, 16, 32, 64, 128]
    
    results_table = []
    
    for batch_size in batch_sizes:
        config = BatchConfig(
            optimal_batch_size=batch_size,
            enable_dynamic_batching=False
        )
        
        optimizer = BatchInferenceOptimizer(analyzer, config)
        
        start_time = time.time()
        results = optimizer.batch_predict(test_comments)
        end_time = time.time()
        
        processing_time = end_time - start_time
        throughput = len(test_comments) / processing_time
        
        stats = optimizer.processor.get_stats()
        
        results_table.append({
            'batch_size': batch_size,
            'time': processing_time,
            'throughput': throughput,
            'avg_batch': stats.get('average_batch_size', 0)
        })
        
        print(f"\nBatch Size: {batch_size}")
        print(f"  Processing Time: {processing_time:.2f}s")
        print(f"  Throughput: {throughput:.1f} comments/sec")
        print(f"  Average Batch Size: {stats.get('average_batch_size', 0):.1f}")
        print(f"  Total Batches: {stats.get('total_batches', 0)}")
    
    # Print summary table
    print("\n" + "=" * 60)
    print("Summary Table")
    print("=" * 60)
    print(f"{'Batch Size':<12} {'Time (s)':<10} {'Throughput':<15} {'Speedup':<10}")
    print("-" * 60)
    
    baseline_time = results_table[0]['time']
    for result in results_table:
        speedup = baseline_time / result['time']
        print(f"{result['batch_size']:<12} {result['time']:<10.2f} "
              f"{result['throughput']:<15.1f} {speedup:<10.2f}x")


def compare_methods():
    """Compare different processing methods."""
    print("\n" + "=" * 60)
    print("Comparing Processing Methods")
    print("=" * 60)
    
    test_comments = generate_test_comments(500)
    
    try:
        analyzer = MLSentimentAnalyzer()
    except Exception as e:
        print(f"Warning: Could not initialize MLSentimentAnalyzer: {e}")
        print("Using mock analyzer for testing")
        
        class MockAnalyzer:
            def analyze_sentiment(self, text):
                time.sleep(0.001)  # Simulate processing time
                return {
                    'sentiment': random.choice(['positive', 'negative', 'neutral']),
                    'confidence': random.random(),
                    'method': 'mock'
                }
            
            def analyze_batch(self, texts):
                return [self.analyze_sentiment(text) for text in texts]
            
            def analyze_batch_optimized(self, texts, **kwargs):
                # Simulate optimized batch processing
                time.sleep(len(texts) * 0.0005)
                return [{'sentiment': random.choice(['positive', 'negative', 'neutral']),
                        'confidence': random.random(),
                        'method': 'mock_optimized'} for _ in texts]
        
        analyzer = MockAnalyzer()
    
    # Method 1: Sequential processing
    print("\n1. Sequential Processing:")
    start_time = time.time()
    sequential_results = []
    for comment in test_comments:
        result = analyzer.analyze_sentiment(comment)
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    
    print(f"  Time: {sequential_time:.2f}s")
    print(f"  Throughput: {len(test_comments)/sequential_time:.1f} comments/sec")
    
    # Method 2: Basic batch processing
    if hasattr(analyzer, 'analyze_batch'):
        print("\n2. Basic Batch Processing:")
        start_time = time.time()
        batch_results = analyzer.analyze_batch(test_comments)
        batch_time = time.time() - start_time
        
        print(f"  Time: {batch_time:.2f}s")
        print(f"  Throughput: {len(test_comments)/batch_time:.1f} comments/sec")
        print(f"  Speedup: {sequential_time/batch_time:.2f}x")
    else:
        batch_time = sequential_time
        print("\n2. Basic Batch Processing: Not available")
    
    # Method 3: Optimized batch processing
    if hasattr(analyzer, 'analyze_batch_optimized'):
        print("\n3. Optimized Batch Processing:")
        start_time = time.time()
        optimized_results = analyzer.analyze_batch_optimized(
            test_comments,
            use_dynamic_batching=True
        )
        optimized_time = time.time() - start_time
        
        print(f"  Time: {optimized_time:.2f}s")
        print(f"  Throughput: {len(test_comments)/optimized_time:.1f} comments/sec")
        print(f"  Speedup vs Sequential: {sequential_time/optimized_time:.2f}x")
        print(f"  Speedup vs Basic Batch: {batch_time/optimized_time:.2f}x")
    else:
        print("\n3. Optimized Batch Processing: Not available")


def test_dynamic_batching():
    """Test dynamic batch sizing based on memory."""
    print("\n" + "=" * 60)
    print("Testing Dynamic Batch Sizing")
    print("=" * 60)
    
    # Generate test data with varying lengths
    short_comments = ["Short " * random.randint(1, 5) for _ in range(100)]
    medium_comments = ["Medium comment " * random.randint(10, 20) for _ in range(100)]
    long_comments = ["This is a very long comment " * random.randint(30, 50) for _ in range(100)]
    
    test_sets = [
        ("Short Comments", short_comments),
        ("Medium Comments", medium_comments),
        ("Long Comments", long_comments)
    ]
    
    for name, comments in test_sets:
        print(f"\n{name}:")
        print(f"  Count: {len(comments)}")
        print(f"  Avg Length: {sum(len(c) for c in comments) / len(comments):.0f} chars")
        
        config = BatchConfig(
            enable_dynamic_batching=True,
            min_batch_size=4,
            max_batch_size=64,
            optimal_batch_size=32
        )
        
        try:
            from app.ml.batch_processor import DynamicBatchProcessor
            processor = DynamicBatchProcessor(config)
            
            text_lengths = [len(c) for c in comments]
            optimal_batch = processor.calculate_optimal_batch_size(text_lengths)
            
            print(f"  Calculated Optimal Batch Size: {optimal_batch}")
            
            batches, _ = processor.create_batches(comments)
            print(f"  Number of Batches Created: {len(batches)}")
            print(f"  Actual Batch Sizes: {[len(b) for b in batches[:5]]}...")
            
        except Exception as e:
            print(f"  Error testing dynamic batching: {e}")


def test_gpu_optimization():
    """Test GPU optimization if available."""
    print("\n" + "=" * 60)
    print("Testing GPU Optimization")
    print("=" * 60)
    
    import torch
    
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Test GPU-optimized processing
        try:
            fast_analyzer = FastSentimentAnalyzer()
            test_comments = generate_test_comments(500)
            
            print("\nGPU Batch Processing:")
            start_time = time.time()
            results = fast_analyzer.analyze_batch_gpu_optimized(test_comments)
            gpu_time = time.time() - start_time
            
            print(f"  Time: {gpu_time:.2f}s")
            print(f"  Throughput: {len(test_comments)/gpu_time:.1f} comments/sec")
            
            # Test with prefetching
            print("\nGPU with Prefetching:")
            start_time = time.time()
            results = fast_analyzer.analyze_with_prefetch(test_comments)
            prefetch_time = time.time() - start_time
            
            print(f"  Time: {prefetch_time:.2f}s")
            print(f"  Throughput: {len(test_comments)/prefetch_time:.1f} comments/sec")
            print(f"  Speedup: {gpu_time/prefetch_time:.2f}x")
            
        except Exception as e:
            print(f"Error testing GPU optimization: {e}")
    else:
        print("GPU not available. CUDA is not installed or no GPU detected.")
        print("To enable GPU optimization:")
        print("  1. Install PyTorch with CUDA support")
        print("  2. Ensure you have a CUDA-capable GPU")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Batch Inference Optimization Test Suite")
    print("=" * 60)
    
    # Run tests
    try:
        benchmark_batch_sizes()
    except Exception as e:
        print(f"\nError in batch size benchmark: {e}")
    
    try:
        compare_methods()
    except Exception as e:
        print(f"\nError in method comparison: {e}")
    
    try:
        test_dynamic_batching()
    except Exception as e:
        print(f"\nError in dynamic batching test: {e}")
    
    try:
        test_gpu_optimization()
    except Exception as e:
        print(f"\nError in GPU optimization test: {e}")
    
    print("\n" + "=" * 60)
    print("Test Suite Completed")
    print("=" * 60)


if __name__ == "__main__":
    main()