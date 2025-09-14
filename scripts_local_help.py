import os
import pathlib
import subprocess

REPO_DIR = pathlib.Path(__file__).resolve().parent

steps = [
    "python -m venv .venv",
    "source .venv/bin/activate && pip install -r requirements.txt",
    "source .venv/bin/activate && pytest -q",
    "modal token new  # follow prompts once",
    "modal deploy modal_app.py",
]

print("Suggested local setup & deploy steps for Modal ML service (run manually):")
for s in steps:
    print("- ", s)
