import os
import types
import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import _normalize_label


class StubRemote:
    def __init__(self, results):
        self._results = results
    def __call__(self, texts):
        # mimic remote returning list of results for batch
        return list(self._results)


class StubPipeline:
    def __init__(self, results):
        self._results = results
    def __call__(self, texts):
        # If results is a callable, generate per text
        if callable(self._results):
            return [self._results(t) for t in texts]
        # Otherwise, return a copy with length matching input
        if len(texts) <= len(self._results):
            return self._results[:len(texts)]
        # Repeat last element if more texts than provided results
        return self._results + [self._results[-1]] * (len(texts) - len(self._results))


@pytest.fixture
def client():
    return TestClient(app_module.app)


def test_normalize_label_variants():
    # Direct labels
    assert _normalize_label("positive") == "positive"
    assert _normalize_label("neutral") == "neutral"
    assert _normalize_label("negative") == "negative"
    # Older RoBERTa LABEL_* variants
    assert _normalize_label("LABEL_2") == "positive"
    assert _normalize_label("LABEL_1") == "neutral"
    assert _normalize_label("LABEL_0") == "negative"
    # Variations
    assert _normalize_label("pos") == "positive"
    assert _normalize_label("neg") == "negative"
    # Unknown defaults to neutral
    assert _normalize_label("something-else") == "neutral"


def test_analyze_text_remote_gpu_path(monkeypatch, client):
    # Force remote GPU path and mock remote call
    monkeypatch.setattr(app_module, "_should_use_remote_gpu", lambda: True)
    monkeypatch.setattr(app_module, "_gpu_analyze_remote", lambda texts: [{"label": "LABEL_2", "score": 0.89}])

    r = client.post("/analyze-text", json={"text": "anything"})
    assert r.status_code == 200
    data = r.json()
    assert data["result"]["predicted_sentiment"] == "positive"
    assert data["result"]["confidence"] == pytest.approx(0.89)


def test_analyze_batch_local_batched(monkeypatch, client):
    # Force local pipeline and provide >32 inputs to exercise batching branch
    monkeypatch.setattr(app_module, "_should_use_remote_gpu", lambda: False)
    # Use a pipeline that returns alternating labels to hit distribution logic
    def make_result(t):
        idx = int(str(t).split()[-1]) if str(t).split() else 0
        if idx % 3 == 0:
            return {"label": "positive", "score": 0.95}
        elif idx % 3 == 1:
            return {"label": "negative", "score": 0.9}
        return {"label": "neutral", "score": 0.5}
    monkeypatch.setattr(app_module, "get_pipeline", lambda: StubPipeline(make_result))

    texts = [f"text {i}" for i in range(65)]  # two batches
    r = client.post("/analyze-batch", json={"texts": texts})
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    assert len(data["results"]) == 65
    assert data["statistics"]["total_analyzed"] == 65
    assert data["overall_sentiment"] in {"positive", "neutral", "negative"}


def test_analyze_batch_error_path(monkeypatch, client):
    # Force error in analysis to hit exception handler
    monkeypatch.setattr(app_module, "_should_use_remote_gpu", lambda: True)
    monkeypatch.setattr(app_module, "_gpu_analyze_remote", lambda texts: (_ for _ in ()).throw(RuntimeError("boom")))

    r = client.post("/analyze-batch", json={"texts": ["a", "b"]})
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is False
    assert "error" in data


def test_summarize_success_and_cache(monkeypatch, client):
    # Stub cache that records set/get
    class FakeCache:
        def __init__(self):
            self.store = {}
        def get(self, prefix, key):
            return self.store.get((prefix, key))
        def set(self, prefix, key, data, ttl_hours=6):
            self.store[(prefix, key)] = data
            return True
    fake_cache = FakeCache()
    monkeypatch.setattr(app_module, "cache", fake_cache)

    # Stub summarizer to avoid real OpenAI call
    class FakeSummarizer:
        def generate_summary(self, comments, sentiment):
            return {"summary": "ok", "comments_analyzed": len(comments), "method": "openai"}
    # Patch the import location used in app.summarize
    import builtins
    real_import = builtins.__import__
    def fake_import(name, *args, **kwargs):
        if name == "app_modules.science.comment_summarizer":
            mod = types.SimpleNamespace(CommentSummarizer=FakeSummarizer)
            return mod
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    payload = {"comments": [{"text": "a", "likes": 1}, {"text": "b", "likes": 2}], "sentiment": {"sentiment_distribution": {"positive":1,"neutral":1,"negative":0}}}
    r1 = client.post("/summarize", json=payload)
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["success"] is True
    assert d1["summary"]["summary"] == "ok"

    # Second call should hit cache and return same
    r2 = client.post("/summarize", json=payload)
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2 == d1


def test_summarize_error_path(monkeypatch, client):
    # Patch to raise during summarizer import/init
    import builtins
    real_import = builtins.__import__
    def fake_import(name, *args, **kwargs):
        if name == "app_modules.science.comment_summarizer":
            class BadSummarizer:
                def __init__(self):
                    raise RuntimeError("no openai")
            return types.SimpleNamespace(CommentSummarizer=BadSummarizer)
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    r = client.post("/summarize", json={"comments": [{"text": "x", "likes": 0}]})
    assert r.status_code == 200
    d = r.json()
    assert d["success"] is False
    assert "OpenAI summarization failed" in d.get("error", "")


def test_ml_service_client_small_branches(monkeypatch):
    # Ensure env var doesn't mask the error path
    monkeypatch.delenv("MODAL_ML_BASE_URL", raising=False)
    # Cover base_url missing error
    from ml_service_client import MLServiceClient
    with pytest.raises(RuntimeError):
        MLServiceClient(base_url="")
    # Cover Authorization header path
    c = MLServiceClient(base_url="http://example.com", api_key="abc")
    headers = c._headers()
    assert headers.get("Authorization") == "Bearer abc"
