from fastapi import FastAPI
from pydantic import BaseModel
import os
import threading
from transformers import pipeline
from typing import List, Dict, Any, Optional
import json
import hashlib

# Cap threads to avoid oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Redis cache (optional)
try:
    from cache import cache
except Exception:
    cache = None  # Cache is optional

MODEL_NAME = os.getenv("MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english")

# Lazy, thread-safe singleton for pipeline
_pipeline = None
_lock = threading.Lock()


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        with _lock:
            if _pipeline is None:
                import torch
                # Automatically use GPU if available, otherwise CPU
                device = 0 if torch.cuda.is_available() else -1
                _pipeline = pipeline(
                    "sentiment-analysis",
                    model=MODEL_NAME,
                    device=device,
                    batch_size=32,
                    truncation=True,
                    max_length=512,
                )
                print(f"Model loaded on device: {'GPU' if device == 0 else 'CPU'}")
    return _pipeline


app = FastAPI(title="Sentiment ML Service")


class TextIn(BaseModel):
    text: str
    method: Optional[str] = "auto"  # auto | ensemble | roberta | fast | ml


class BatchIn(BaseModel):
    texts: List[str]
    method: Optional[str] = "auto"


class SummarizeIn(BaseModel):
    comments: List[Dict[str, Any]]
    sentiment: Optional[Dict[str, Any]] = None
    method: Optional[str] = "auto"  # auto | transformer | objective | openai


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


def _normalize_label(label: str) -> str:
    label = (label or "neutral").lower()
    if "pos" in label:
        return "positive"
    if "neg" in label:
        return "negative"
    if label in ("positive", "neutral", "negative"):
        return label
    return "neutral"


@app.post("/analyze-text")
def analyze_text(body: TextIn):
    # For now, we use the transformers pipeline; future: route by body.method
    pipe = get_pipeline()
    res = pipe([body.text])[0]
    label = _normalize_label(res.get("label", "neutral"))
    return {
        "success": True,
        "result": {
            "predicted_sentiment": label,
            "confidence": float(res.get("score", 0.0)),
        },
        "models_used": ["distilbert"],
        "method": body.method or "auto",
    }


@app.post("/analyze-batch")
def analyze_batch(body: BatchIn):
    if not body.texts:
        return {
            "success": True, 
            "results": [], 
            "statistics": {
                "total_analyzed": 0,
                "sentiment_distribution": {"positive": 0, "neutral": 0, "negative": 0},
                "sentiment_percentages": {"positive": 0.0, "neutral": 0.0, "negative": 0.0},
                "average_confidence": 0.0
            },
            "overall_sentiment": "neutral"
        }

    # Use the pipeline directly for all methods - simpler and more reliable
    try:
        pipe = get_pipeline()
        
        # Process all texts at once if small batch, otherwise chunk it
        texts_list = list(body.texts)  # Ensure it's a list
        if len(texts_list) <= 32:
            raw_results = pipe(texts_list)
        else:
            # Process in batches if needed for better performance
            batch_size = 32
            raw_results = []
            for i in range(0, len(texts_list), batch_size):
                batch_texts = texts_list[i:i+batch_size]
                batch_raw = pipe(batch_texts)
                raw_results.extend(batch_raw)
        
        results = []
        counts = {"positive": 0, "neutral": 0, "negative": 0}
        confidence_sum = 0.0
        
        for text, res in zip(texts_list, raw_results):
            label = _normalize_label(res.get("label", "neutral"))
            conf = float(res.get("score", 0.0))
            results.append({
                "text": str(text)[:100],  # Ensure it's a string
                "predicted_sentiment": label, 
                "confidence": conf
            })
            counts[label] += 1
            confidence_sum += conf
        
        total = len(results)
        avg_confidence = confidence_sum / total if total > 0 else 0.0
        
        # Calculate percentages
        percentages = {
            "positive": (counts["positive"] / total * 100.0) if total > 0 else 0.0,
            "neutral": (counts["neutral"] / total * 100.0) if total > 0 else 0.0,
            "negative": (counts["negative"] / total * 100.0) if total > 0 else 0.0
        }
        
        # Determine overall sentiment
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
                "sentiment_percentages": percentages,
                "average_confidence": avg_confidence
            },
            "overall_sentiment": overall,
            "method": "distilbert",
            "models_used": ["distilbert"]
        }
    except Exception as e:
        print(f"Error in analyze_batch: {e}")
        # Return error response
        return {
            "success": False,
            "error": str(e),
            "results": [],
            "statistics": {
                "total_analyzed": 0,
                "sentiment_distribution": {"positive": 0, "neutral": 0, "negative": 0},
                "sentiment_percentages": {"positive": 0.0, "neutral": 0.0, "negative": 0.0},
                "average_confidence": 0.0
            },
            "overall_sentiment": "neutral"
        }


@app.post("/summarize")
def summarize(body: SummarizeIn):
    # Optional cache
    cache_key = None
    if cache:
        # Hash first 10 comments text + distribution + method
        payload = {
            "texts": [c.get("text", "") for c in body.comments[:10]],
            "dist": (body.sentiment or {}).get("sentiment_distribution")
                    or (body.sentiment or {}).get("sentiment_counts"),
            "method": body.method or "auto",
        }
        key_src = json.dumps(payload, sort_keys=True).encode("utf-8")
        cache_key = hashlib.sha256(key_src).hexdigest()
        cached = cache.get("summary", cache_key)
        if cached:
            return {"success": True, "summary": cached}

    # Import summarizer lazily
    try:
        from app.science.comment_summarizer import CommentSummarizer  # type: ignore
        use_openai = (body.method == "openai") and bool(os.getenv("OPENAI_API_KEY"))
        summarizer = CommentSummarizer(use_openai=use_openai)
        if body.method == "transformer":
            result = summarizer.summarize_with_transformer(body.comments, body.sentiment)
        elif body.method == "objective":
            result = summarizer.create_objective_summary(body.comments, body.sentiment)
        else:
            result = summarizer.generate_summary(body.comments, body.sentiment)
    except Exception as e:
        # Fallback summary
        dist = (body.sentiment or {}).get("sentiment_distribution") or (body.sentiment or {}).get("sentiment_counts") or {}
        pos = dist.get("positive", 0); neu = dist.get("neutral", 0); neg = dist.get("negative", 0)
        total = (body.sentiment or {}).get("total_analyzed", 0) or (pos + neu + neg)
        def pct(x):
            return round((x / total * 100), 1) if total else 0.0
        trend = "mixed"
        if pos >= max(neg, neu) and pos >= total * 0.5:
            trend = "mostly positive"
        elif neg >= max(pos, neu) and neg >= total * 0.4:
            trend = "mostly negative"
        result = {
            "summary": (
                f"Viewer reactions are {trend}. Distribution — "
                f"positive: {pct(pos)}%, neutral: {pct(neu)}%, negative: {pct(neg)}%."
            ),
            "method": "fallback",
            "comments_analyzed": total,
        }

    if cache and cache_key:
        cache.set("summary", cache_key, result, ttl_hours=6)

    return {"success": True, "summary": result}
