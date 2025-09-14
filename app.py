from fastapi import FastAPI
from pydantic import BaseModel
import os
import threading
from transformers import pipeline

# Cap threads to avoid oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_NAME = os.getenv("MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english")

# Lazy, thread-safe singleton for pipeline
_pipeline = None
_lock = threading.Lock()


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        with _lock:
            if _pipeline is None:
                _pipeline = pipeline("sentiment-analysis", model=MODEL_NAME, device=-1, batch_size=32, truncation=True)
    return _pipeline


app = FastAPI(title="Sentiment ML Service")


class TextIn(BaseModel):
    text: str
    method: str | None = "auto"


class BatchIn(BaseModel):
    texts: list[str]
    method: str | None = "auto"


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/analyze-text")
def analyze_text(body: TextIn):
    pipe = get_pipeline()
    res = pipe([body.text])[0]
    label = res.get("label", "neutral").lower()
    if "pos" in label:
        label = "positive"
    elif "neg" in label:
        label = "negative"
    else:
        label = "neutral"
    return {
        "success": True,
        "result": {
            "predicted_sentiment": label,
            "confidence": float(res.get("score", 0.0))
        },
        "models_used": ["distilbert"],
    }


@app.post("/analyze-batch")
def analyze_batch(body: BatchIn):
    if not body.texts:
        return {"success": True, "results": [], "statistics": {"total_analyzed": 0}}
    pipe = get_pipeline()
    raw = pipe(body.texts)
    results = []
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for t, r in zip(body.texts, raw):
        label = r.get("label", "neutral").lower()
        if "pos" in label:
            label = "positive"
        elif "neg" in label:
            label = "negative"
        else:
            label = "neutral"
        results.append({
            "text": t[:100],
            "predicted_sentiment": label,
            "confidence": float(r.get("score", 0.0))
        })
        counts[label] += 1
    total = len(results)
    if counts["positive"] >= total * 0.5:
        overall = "positive"
    elif counts["negative"] >= total * 0.4:
        overall = "negative"
    else:
        overall = "neutral"
    return {
        "success": True,
        "results": results,
        "statistics": {
            "total_analyzed": total,
            "sentiment_distribution": counts,
        },
        "overall_sentiment": overall,
    }
