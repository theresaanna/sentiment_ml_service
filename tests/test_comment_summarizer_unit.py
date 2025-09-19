import os
import pytest

from app_modules.science import comment_summarizer as cs


def make_instance_without_init():
    # Create an instance without calling __init__ to avoid OpenAI dependency
    inst = cs.CommentSummarizer.__new__(cs.CommentSummarizer)
    return inst


def test_init_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        cs.CommentSummarizer()


def test_prepare_comments_text_with_sentiment():
    inst = make_instance_without_init()
    comments = [
        {"text": "Great video!", "likes": 10},
        {"text": "Terrible audio.", "likes": 1},
        {"text": "Okay content.", "likes": 2},
    ]
    sentiment = {
        "individual_results": [
            {"predicted_sentiment": "positive"},
            {"predicted_sentiment": "negative"},
            {"predicted_sentiment": "neutral"},
        ]
    }
    txt = inst._prepare_comments_text(comments, sentiment)
    assert "Supportive viewpoints:" in txt
    assert "Critical perspectives:" in txt
    assert "Neutral observations:" in txt


def test_prepare_comments_text_no_sentiment_truncation():
    inst = make_instance_without_init()
    comments = [{"text": f"comment {i}"} for i in range(40)]  # > 30, will be truncated to first 30 and <= 4000 chars
    txt = inst._prepare_comments_text(comments, None)
    assert "comment 0" in txt
    assert len(txt) <= 4000


def test_extract_themes_and_engagement_metrics():
    inst = make_instance_without_init()
    comments = [
        {"text": "Love the camera work and editing", "likes": 5},
        {"text": "Editing was superb, loved the editing style", "likes": 8},
        {"text": "Camera quality could be better", "likes": 2, "is_reply": True},
        {"text": "The content is informative and engaging", "likes": 7},
    ]
    themes = inst._extract_themes(comments, top_n=3)
    assert isinstance(themes, list) and len(themes) <= 3
    # Likely to include 'editing' from repeated mentions
    assert any(t in ("editing", "camera", "content") for t in themes)

    metrics = inst._calculate_engagement_metrics(comments)
    assert metrics["total_likes"] == sum(c.get("likes", 0) for c in comments)
    assert metrics["most_liked_count"] == 8
    assert metrics["reply_rate"] >= 0.0