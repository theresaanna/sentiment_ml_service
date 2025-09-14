import modal
from modal import App, Image, asgi_app, gpu
import sys
import os

# GPU-enabled image with CUDA support and dependencies
image = (
    Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("pytest", "httpx")  # Add test dependencies
    .env({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "CUDA_VISIBLE_DEVICES": "0",  # Use first GPU
    })
    .add_local_file("app.py", "/root/app.py")  # Add app.py last to avoid rebuilding
    .add_local_dir("tests", "/root/tests")  # Add tests directory
)

app = App("sentiment-ml-service")

# Test runner function
@app.function(
    image=image,
    gpu=gpu.T4(),  # Use T4 GPU for tests (cost-effective)
    timeout=300,
)
def run_tests():
    """Run tests in Modal environment with GPU."""
    import subprocess
    import sys
    
    # Add root to path for imports
    sys.path.insert(0, "/root")
    
    # Run pytest with verbose output
    result = subprocess.run(
        ["python", "-m", "pytest", "/root/tests/", "-v", "--tb=short"],
        capture_output=True,
        text=True,
        cwd="/root"
    )
    
    print("Test Output:")
    print(result.stdout)
    if result.stderr:
        print("Test Errors:")
        print(result.stderr)
    
    if result.returncode != 0:
        raise Exception(f"Tests failed with return code {result.returncode}")
    
    return "All tests passed successfully!"

# Main FastAPI app with GPU support
@app.function(
    image=image,
    gpu=gpu.T4(),  # Use T4 GPU for inference
    min_containers=1,  # Keep at least one container warm
    max_containers=10,  # Auto-scale up to 10 containers
    timeout=120,
    allow_concurrent_inputs=100,  # Handle multiple requests concurrently
)
@asgi_app()
def fastapi_app():
    import sys
    sys.path.insert(0, "/root")
    from app import app as fastapi_instance
    return fastapi_instance

# Health check function
@app.function(image=image, gpu=gpu.T4())
def health_check():
    """Check if GPU is available and model can be loaded."""
    import torch
    import sys
    sys.path.insert(0, "/root")
    from app import get_pipeline
    
    # Check GPU availability
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0
    
    # Try to load the model
    try:
        pipeline = get_pipeline()
        model_loaded = True
    except Exception as e:
        model_loaded = False
        error = str(e)
    
    return {
        "gpu_available": gpu_available,
        "gpu_count": gpu_count,
        "model_loaded": model_loaded,
        "error": error if not model_loaded else None
    }

if __name__ == "__main__":
    # Run tests before deploying
    with app.run():
        print("Running tests...")
        result = run_tests.remote()
        print(result)
    
    # Deploy the app
    print("Deploying app...")
    app.deploy("sentiment-ml-service")
