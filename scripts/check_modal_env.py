import os
import json

MODAL_ML_BASE_URL = os.getenv("MODAL_ML_BASE_URL", "")

print("Modal ML integration status:")
if MODAL_ML_BASE_URL:
    print(f"- Base URL configured: {MODAL_ML_BASE_URL}")
else:
    print("- MODAL_ML_BASE_URL is not set. Set it to your Modal web app URL.")

print("Suggested environment variables:")
print("- MODAL_ML_BASE_URL=https://sentiment-ml-service--<handle>.modal.run")
print("- PRECOMPUTE_ANALYSIS_ON_PRELOAD=true  # optional")
print("- PRELOAD_ANALYSIS_LIMIT=500  # optional")
print("- PRELOAD_ANALYSIS_METHOD=auto  # optional")
