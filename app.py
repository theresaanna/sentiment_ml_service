from fastapi import FastAPI
from pydantic import BaseModel
import os
import threading
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

# Use RoBERTa model that supports 3-class sentiment (positive, neutral, negative)
MODEL_NAME = os.getenv("MODEL_NAME", "cardiffnlp/twitter-roberta-base-sentiment-latest")

# Lazy, thread-safe singleton for pipeline
_pipeline = None
_lock = threading.Lock()


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        with _lock:
            if _pipeline is None:
                # Optional lightweight rule-based pipeline for constrained/test envs
                if os.getenv("USE_FAKE_PIPELINE") == "1":
                    class _FakeSentimentPipeline:
                        def __call__(self, texts):
                            if isinstance(texts, str):
                                texts = [texts]
                            results = []
                            for t in texts:
                                s = (t or "").lower()
                                score = 0.5
                                label = "neutral"
                                pos_keywords = ["love", "amazing", "fantastic", "best", "great", "wonderful", "😊", "❤️", "highly recommend", "awesome"]
                                neg_keywords = ["terrible", "hate", "worst", "disappointed", "awful", "bad", "can't stand", "horrible"]
                                if any(k in s for k in pos_keywords):
                                    label, score = "positive", 0.95
                                if any(k in s for k in neg_keywords):
                                    label, score = "negative", 0.9
                                results.append({"label": label, "score": float(score)})
                            return results
                    _pipeline = _FakeSentimentPipeline()
                else:
                    # Heavy pipeline only when explicitly needed
                    import torch  # type: ignore
                    from transformers import pipeline  # type: ignore
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


# Remote GPU analysis integration
MODAL_APP_NAME = os.getenv("MODAL_APP_NAME", "sentiment-ml-service")
_GPU_FN = None


def _should_use_remote_gpu() -> bool:
    # Use remote GPU when not explicitly running the fake pipeline (tests/stubs)
    return os.getenv("USE_FAKE_PIPELINE") != "1"


def _gpu_analyze_remote(texts: List[str]) -> List[Dict[str, Any]]:
    """Call the Modal GPU function to analyze texts.

    Returns a list of {label, score}.
    """
    global _GPU_FN
    if _GPU_FN is None:
        try:
            from modal import Function as ModalFunction  # Lazy import for local/test runs
        except Exception as e:
            raise RuntimeError(f"Modal client not available: {e}")
        _GPU_FN = ModalFunction.from_name(MODAL_APP_NAME, "gpu_analyze_batch")
    return _GPU_FN.remote(list(texts))


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
    video_title: Optional[str] = None  # Video title to filter out redundant words
    method: Optional[str] = "auto"  # auto | transformer | objective | openai


@app.get("/health")
def health():
    # Report OpenAI configuration status without exposing secrets
    openai_key_present = bool(os.getenv("OPENAI_API_KEY"))
    openai_model = os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4o-mini")
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "openai": {
            "configured": openai_key_present,
            "model": openai_model,
        },
    }


def _normalize_label(label: str) -> str:
    """Normalize sentiment labels from RoBERTa model.
    
    RoBERTa twitter-roberta-base-sentiment-latest outputs: 'positive', 'neutral', 'negative'
    (Some older versions might output LABEL_0/1/2)
    """
    label = (label or "neutral").lower()
    
    # Direct text labels (what RoBERTa actually outputs)
    if label in ("positive", "neutral", "negative"):
        return label
    
    # Handle older RoBERTa label format (just in case)
    if "label_0" in label:
        return "negative"
    if "label_1" in label:
        return "neutral"
    if "label_2" in label:
        return "positive"
    
    # Handle variations
    if "pos" in label:
        return "positive"
    if "neg" in label:
        return "negative"
    
    # Default to neutral if unknown
    return "neutral"


@app.post("/analyze-text")
def analyze_text(body: TextIn):
    # Route analysis to a GPU-backed Modal function by default (except in fake/test mode)
    if _should_use_remote_gpu():
        res = _gpu_analyze_remote([body.text])[0]
    else:
        pipe = get_pipeline()
        res = pipe([body.text])[0]

    label = _normalize_label(res.get("label", "neutral"))
    return {
        "success": True,
        "result": {
            "predicted_sentiment": label,
            "confidence": float(res.get("score", 0.0)),
        },
        "models_used": ["roberta"],
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

    try:
        # Normalize input
        texts_list = list(body.texts)

        # Route to GPU-backed function unless in fake/test mode
        if _should_use_remote_gpu():
            raw_results = _gpu_analyze_remote(texts_list)
        else:
            pipe = get_pipeline()
            if len(texts_list) <= 32:
                raw_results = pipe(texts_list)
            else:
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
                "text": str(text)[:100],
                "predicted_sentiment": label,
                "confidence": conf
            })
            counts[label] += 1
            confidence_sum += conf

        total = len(results)
        avg_confidence = confidence_sum / total if total > 0 else 0.0

        percentages = {
            "positive": (counts["positive"] / total * 100.0) if total > 0 else 0.0,
            "neutral": (counts["neutral"] / total * 100.0) if total > 0 else 0.0,
            "negative": (counts["negative"] / total * 100.0) if total > 0 else 0.0
        }

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
            "method": "roberta",
            "models_used": ["roberta"]
        }
    except Exception as e:
        print(f"Error in analyze_batch: {e}")
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
        from app_modules.science.comment_summarizer import CommentSummarizer  # type: ignore
        summarizer = CommentSummarizer()
        result = summarizer.generate_summary(body.comments, body.sentiment)
    except Exception as e:
        return {"success": False, "error": f"OpenAI summarization failed: {str(e)}"}

    if cache and cache_key:
        cache.set("summary", cache_key, result, ttl_hours=6)

    return {"success": True, "summary": result}
