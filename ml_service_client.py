import os
from typing import Any, Dict, List, Optional
import httpx

class MLServiceClient:
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, timeout: float = 20.0):
        self.base_url = (base_url or os.getenv("MODAL_ML_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("MODAL_ML_API_KEY")
        self.timeout = timeout
        if not self.base_url:
            raise RuntimeError("MODAL_ML_BASE_URL not configured")

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def analyze_text(self, text: str, method: str = "auto") -> Dict[str, Any]:
        url = f"{self.base_url}/analyze-text"
        payload = {"text": text, "method": method}
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(url, json=payload, headers=self._headers())
            r.raise_for_status()
            data = r.json()
        # Normalize to the schema used by unified analyzer
        result = data.get("result") or {}
        return {
            "predicted_sentiment": result.get("predicted_sentiment", "neutral"),
            "confidence": float(result.get("confidence", 0.0)),
            "method": method,
            "models_used": data.get("models_used", []),
        }

    def analyze_batch(self, texts: List[str], method: str = "auto") -> Dict[str, Any]:
        url = f"{self.base_url}/analyze-batch"
        payload = {"texts": texts, "method": method}
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(url, json=payload, headers=self._headers())
            r.raise_for_status()
            data = r.json()
        results = data.get("results", [])
        # Normalize per-item keys to what your templates expect
        individual_results = []
        for item in results:
            individual_results.append({
                "text": item.get("text"),
                "predicted_sentiment": item.get("predicted_sentiment", item.get("label", "neutral")),
                "confidence": float(item.get("confidence", 0.0)),
                "sentiment_scores": {},
            })
        stats = data.get("statistics", {})
        # Build a distribution compatible with existing code
        dist = stats.get("sentiment_distribution") or {"positive": 0, "neutral": 0, "negative": 0}
        total = stats.get("total_analyzed", len(individual_results))
        pct = {k: (v / total * 100 if total else 0.0) for k, v in dist.items()}
        return {
            "results": individual_results,
            "statistics": {
                "total_analyzed": total,
                "sentiment_distribution": dist,
                "sentiment_percentages": pct,
                "average_confidence": sum(x.get("confidence", 0.0) for x in individual_results) / total if total else 0.0,
            },
            "method": method,
            "total_analyzed": total,
        }
