import modal
from modal import App, Image, asgi_app

# Image with dependencies from requirements.txt
image = (
    Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .env({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    .add_local_file("app.py", "/root/app.py")  # Add app.py last to avoid rebuilding
)

app = App("sentiment-ml-service")


@app.function(
    image=image, 
    min_containers=1,  # Using min_containers instead of keep_warm
    timeout=120
)
@asgi_app()
def fastapi_app():
    import sys
    sys.path.insert(0, "/root")
    from app import app as fastapi_instance
    return fastapi_instance
