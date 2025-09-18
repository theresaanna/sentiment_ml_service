import modal
from modal import App, Image, asgi_app, gpu, Function, Secret
import sys
import os
import argparse

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
    """GPU readiness check: ensures CUDA is visible, loads model, and runs tiny inference.

    Returns basic GPU info plus whether model load and inference succeeded.
    """
    import os
    import sys

    # Ensure GPU is visible even though the base image hides CUDA by default
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Ensure we use the real pipeline here (not the fake one)
    os.environ["USE_FAKE_PIPELINE"] = "0"

    # Imports after setting CUDA visibility
    import torch

    sys.path.insert(0, "/root")
    from service_main import get_pipeline

    # GPU info
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0
    device_name = torch.cuda.get_device_name(0) if gpu_available and gpu_count > 0 else None

    # Try to load the model and run a tiny inference
    model_loaded = False
    inference_ok = False
    result_sample = None
    error = None
    try:
        pipe = get_pipeline()
        model_loaded = True
        # Tiny inference to validate end-to-end path
        sample = ["ok"]
        out = pipe(sample)
        if isinstance(out, list) and len(out) == 1 and isinstance(out[0], dict):
            inference_ok = True
            # Keep only minimal info
            result_sample = {"label": out[0].get("label"), "score": float(out[0].get("score", 0.0))}
    except Exception as e:
        error = str(e)

    return {
        "gpu_available": gpu_available,
        "gpu_count": gpu_count,
        "device_name": device_name,
        "model_loaded": model_loaded,
        "inference_ok": inference_ok,
        "result_sample": result_sample,
        "error": error,
    }


# GPU-backed analysis function
@app.function(
    image=image,
    gpu="T4",
    timeout=300,
    secrets=[Secret.from_name("openai-api-key")],
)
def gpu_analyze_batch(texts: list[str]):
    """Run sentiment analysis on GPU for a list of texts.

    Ensures GPU visibility even if the base image disables CUDA by default.
    Returns a list of dicts with keys: label, score.
    """
    import os
    import sys

    # Ensure GPU is visible within this function even if image env hid it
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Avoid using fake pipeline in production GPU path unless explicitly requested
    os.environ["USE_FAKE_PIPELINE"] = os.environ.get("USE_FAKE_PIPELINE", "0")

    # Make sure our modules are importable
    sys.path.insert(0, "/root")

    from service_main import get_pipeline  # Lazy import to honor env overrides

    # Normalize input
    if isinstance(texts, str):
        texts = [texts]
    texts = list(texts or [])

    # Load once per container; get_pipeline handles internal caching
    pipe = get_pipeline()

    # Batch processing for efficiency
    batch_size = 32
    raw_results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        raw_results.extend(pipe(batch))

    # Map to a simple, stable schema
    results = []
    for r in raw_results:
        try:
            label = r.get("label", "neutral")
            score = float(r.get("score", 0.0))
        except Exception:
            label, score = "neutral", 0.0
        results.append({"label": label, "score": score})

    return results


def main(argv=None):
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
