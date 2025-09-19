from cache import CacheService


def test_cache_service_disabled_paths(monkeypatch):
    # Force redis library to be unavailable
    import cache as cache_mod
    monkeypatch.setattr(cache_mod, "redis", None)

    svc = CacheService(redis_url="redis://localhost:6379/0")
    # Disabled by default without redis lib
    assert svc.enabled is False
    assert svc.redis_client is None

    # Methods should safely no-op
    assert svc.get("p", "k") is None
    assert svc.set("p", "k", {"a": 1}) is False
    assert svc.delete("p", "k") is False
    assert isinstance(svc.clear_pattern("*"), int) and svc.clear_pattern("*") >= 0
    stats = svc.get_cache_stats()
    assert stats.get("enabled") is False

    # Internal key maker
    assert svc._make_key("s", "id") == "mlsvc:s:id"