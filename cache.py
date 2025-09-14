"""
Redis cache service for the sentiment ML service.

This mirrors the CacheService used in the Flask app, adapted to use a
more generic key prefix. Set REDIS_URL to enable.
"""
import os
import json
from typing import Optional, Any
from datetime import timedelta

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # type: ignore


class CacheService:
    """Service for caching results using Redis."""

    def __init__(self, redis_url: Optional[str] = None):
        """Initialize Redis connection."""
        if not redis_url:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

        # TTL settings
        self.default_ttl_hours = int(os.getenv("REDIS_CACHE_TTL_HOURS", "24"))
        self.analysis_ttl_hours = int(os.getenv("REDIS_ANALYSIS_TTL_HOURS", "6"))

        self.enabled = False
        self.redis_client = None

        if redis is None:
            print("Redis library not installed; cache disabled.")
            return

        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)  # type: ignore
            self.redis_client.ping()
            self.enabled = True
            print(f"Redis cache connected successfully to {redis_url}")
            print(
                f"Cache TTL: default={self.default_ttl_hours}h, analysis={self.analysis_ttl_hours}h"
            )
        except Exception as e:
            print(f"Redis connection failed: {e}. Running without cache.")
            self.redis_client = None
            self.enabled = False

    def _make_key(self, prefix: str, identifier: str) -> str:
        """Create a cache key."""
        return f"mlsvc:{prefix}:{identifier}"

    def get(self, prefix: str, identifier: str) -> Optional[Any]:
        """Get cached data."""
        if not self.enabled or not self.redis_client:
            return None
        try:
            key = self._make_key(prefix, identifier)
            data = self.redis_client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:  # pragma: no cover
            print(f"Cache get error: {e}")
        return None

    def set(self, prefix: str, identifier: str, data: Any, ttl_hours: int = 24) -> bool:
        """Set cached data with TTL."""
        if not self.enabled or not self.redis_client:
            return False
        try:
            key = self._make_key(prefix, identifier)
            serialized = json.dumps(data)
            result = self.redis_client.setex(key, timedelta(hours=ttl_hours), serialized)
            return bool(result)
        except Exception as e:  # pragma: no cover
            print(f"Cache set error: {e}")
            return False

    def delete(self, prefix: str, identifier: str) -> bool:
        if not self.enabled or not self.redis_client:
            return False
        try:
            key = self._make_key(prefix, identifier)
            return bool(self.redis_client.delete(key))
        except Exception as e:  # pragma: no cover
            print(f"Cache delete error: {e}")
            return False

    def clear_pattern(self, pattern: str) -> int:
        if not self.enabled or not self.redis_client:
            return 0
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return int(self.redis_client.delete(*keys))
            return 0
        except Exception as e:  # pragma: no cover
            print(f"Cache clear pattern error: {e}")
            return 0

    def get_cache_stats(self) -> dict:
        if not self.enabled or not self.redis_client:
            return {"enabled": False}
        try:
            info = self.redis_client.info("stats")
            keys = int(self.redis_client.dbsize())
            hits = int(info.get("keyspace_hits", 0))
            misses = int(info.get("keyspace_misses", 0))
            return {
                "enabled": True,
                "total_keys": keys,
                "hits": hits,
                "misses": misses,
                "hit_rate": round(hits / (hits + misses + 1e-9) * 100, 2),
            }
        except Exception as e:  # pragma: no cover
            print(f"Cache stats error: {e}")
            return {"enabled": False, "error": str(e)}


# Global cache instance
cache = CacheService()
