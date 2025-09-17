import modal
from modal import App, Image, asgi_app, gpu, Function, Secret
import sys
import os

# Base image with dependencies
image = (
    Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("pytest", "httpx")  # Add test dependencies
    .env({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        # Disable CUDA by default in this image to avoid GPU-related issues during tests
        "CUDA_VISIBLE_DEVICES": "",
        # Configure OpenAI model for summaries (can be overridden at deploy time)
        "OPENAI_SUMMARY_MODEL": os.environ.get("OPENAI_SUMMARY_MODEL", "gpt-4o-mini"),
    })
    # Include the full app_modules package (analyzers and model artifacts)
    .add_local_dir("app_modules", "/root/app_modules")
    # Include FastAPI entrypoint and cache shim
    .add_local_file("app.py", "/root/service_main.py")
    .add_local_file("app.py", "/root/app.py")
    .add_local_file("cache.py", "/root/cache.py")
    # Add tests directory for CI and remote testing
    .add_local_dir("tests", "/root/tests")
)

APP_NAME = "sentiment-ml-service"
app = App(APP_NAME)

# Test runner function
@app.function(
    image=image,
    timeout=300,
    secrets=[Secret.from_name("openai-api-key")],
)
def run_tests(pipeline_mode: str = "auto"):
    """Run tests in Modal environment (CPU-only for stability).

    pipeline_mode: one of {"auto", "fake", "real"}
    """
    import subprocess
    import sys
    import os
    
    # Force CPU execution to avoid GPU driver/toolkit mismatches during tests
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Add root to path for imports
    sys.path.insert(0, "/root")

    # Run pytest with verbose output, configuring the pipeline mode
    env = os.environ.copy()
    if pipeline_mode == "real":
        env["USE_FAKE_PIPELINE"] = "0"
    elif pipeline_mode == "fake":
        env["USE_FAKE_PIPELINE"] = "1"
    else:  # auto
        env["USE_FAKE_PIPELINE"] = env.get("USE_FAKE_PIPELINE", "1")

    result = subprocess.run(
        ["python", "-m", "pytest", "/root/tests/", "-v", "--tb=short"],
        capture_output=True,
        text=True,
        cwd="/root",
        env=env,
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
    # Keep inference CPU-first; the model is small and CPU is fine
    min_containers=1,  # Keep at least one container warm
    max_containers=10,  # Auto-scale up to 10 containers
    timeout=120,
    secrets=[Secret.from_name("openai-api-key")],
)
@asgi_app()
def fastapi_app():
    import sys
    import os
    # Prefer CPU in service too, unless Modal injects a GPU runtime
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Honor USE_FAKE_PIPELINE if provided by environment, but don't force it here
    sys.path.insert(0, "/root")
    from service_main import app as fastapi_instance
    return fastapi_instance

# Health check function
@app.function(image=image, gpu="T4")
def health_check():
    """Check if GPU is available and model can be loaded."""
    import torch
    import sys
    sys.path.insert(0, "/root")
    from service_main import get_pipeline
    
    # Check GPU availability
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0
    
    # Try to load the model
    try:
        pipeline = get_pipeline()
        model_loaded = True
        error = None
    except Exception as e:
        model_loaded = False
        error = str(e)
    
    return {
        "gpu_available": gpu_available,
        "gpu_count": gpu_count,
        "model_loaded": model_loaded,
        "error": error
    }

def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        description="Deploy/redeploy the existing Modal app without creating new instances."
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="deploy",
        choices=["deploy", "redeploy", "health"],
        help="Action to execute. 'deploy' and 'redeploy' both update the existing deployment. 'health' calls the deployed health_check.",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run the deployed run_tests function after a successful deploy.",
    )
    parser.add_argument(
        "--test-pipeline",
        choices=["auto", "fake", "real"],
        default="auto",
        help="Pipeline mode to use when running tests: auto (default), fake, or real.",
    )
    args = parser.parse_args(argv)

    if args.command in {"deploy", "redeploy"}:
        print(f"Deploying '{APP_NAME}' (this updates the existing deployment)...")
        app.deploy()
        print("Deploy complete.")
        if args.run_tests:
            print("Running tests via deployed function...")
            try:
                tests_fn = Function.from_name(APP_NAME, "run_tests")
                result = tests_fn.remote(args.test_pipeline)
                print(result)
            except Exception as e:
                print(f"Tests failed: {e}")
                sys.exit(1)
    elif args.command == "health":
        print("Checking deployed health...")
        health_fn = Function.from_name(APP_NAME, "health_check")
        info = health_fn.remote()
        print(info)


if __name__ == "__main__":
    main()
