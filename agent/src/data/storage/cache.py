from __future__ import annotations
"""
Redis cache for fast feature access.
"""
import json
from datetime import datetime
from typing import Any
import redis.asyncio as redis
import structlog

from ...config import settings

logger = structlog.get_logger()


class FeatureCache:
    """
    Redis cache for ML features.
    
    Provides sub-5ms latency for feature access during inference.
    
    Cached data:
    - Latest computed features
    - Gas prices
    - Price spreads
    - Model predictions
    """
    
    # Cache TTL in seconds
    DEFAULT_TTL = 300  # 5 minutes
    GAS_TTL = 60       # 1 minute for gas (changes frequently)
    FEATURE_TTL = 300  # 5 minutes for features
    
    def __init__(self, redis_url: str | None = None):
        self.redis_url = redis_url or settings.redis_url
        self.client: redis.Redis | None = None
        self.logger = logger.bind(component="feature_cache")
    
    async def connect(self) -> None:
        """Connect to Redis."""
        if not self.redis_url:
            self.logger.warning("No Redis URL configured")
            return
        
        self.client = redis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True
        )
        
        # Test connection
        try:
            await self.client.ping()
            self.logger.info("Redis connected")
        except Exception as e:
            self.logger.error("Redis connection failed", error=str(e))
            self.client = None
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self.client:
            await self.client.close()
            self.logger.info("Redis connection closed")
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: int = DEFAULT_TTL
    ) -> bool:
        """Set a value in cache."""
        if not self.client:
            return False
        
        try:
            serialized = json.dumps(value, default=str)
            await self.client.setex(key, ttl, serialized)
            return True
        except Exception as e:
            self.logger.error("Cache set failed", key=key, error=str(e))
            return False
    
    async def get(self, key: str) -> Any | None:
        """Get a value from cache."""
        if not self.client:
            return None
        
        try:
            value = await self.client.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            self.logger.error("Cache get failed", key=key, error=str(e))
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        if not self.client:
            return False
        
        try:
            await self.client.delete(key)
            return True
        except Exception as e:
            self.logger.error("Cache delete failed", key=key, error=str(e))
            return False
    
    # Specialized methods for features
    
    async def cache_gas_price(self, gas_data: dict[str, Any]) -> bool:
        """Cache current gas price."""
        return await self.set("gas:current", gas_data, self.GAS_TTL)
    
    async def get_gas_price(self) -> dict[str, Any] | None:
        """Get cached gas price."""
        return await self.get("gas:current")
    
    async def cache_features(
        self,
        strategy: str,
        features: dict[str, Any]
    ) -> bool:
        """Cache computed features for a strategy."""
        key = f"features:{strategy}:latest"
        return await self.set(key, features, self.FEATURE_TTL)
    
    async def get_features(self, strategy: str) -> dict[str, Any] | None:
        """Get cached features for a strategy."""
        key = f"features:{strategy}:latest"
        return await self.get(key)
    
    async def cache_spread(
        self,
        pair: str,
        spread_data: dict[str, Any]
    ) -> bool:
        """Cache price spread for a pair."""
        key = f"spread:{pair}"
        return await self.set(key, spread_data, self.DEFAULT_TTL)
    
    async def get_spread(self, pair: str) -> dict[str, Any] | None:
        """Get cached spread for a pair."""
        key = f"spread:{pair}"
        return await self.get(key)
    
    async def cache_prediction(
        self,
        strategy: str,
        prediction: dict[str, Any]
    ) -> bool:
        """Cache latest prediction."""
        key = f"prediction:{strategy}:latest"
        return await self.set(key, prediction, self.DEFAULT_TTL)
    
    async def get_prediction(self, strategy: str) -> dict[str, Any] | None:
        """Get cached prediction."""
        key = f"prediction:{strategy}:latest"
        return await self.get(key)
