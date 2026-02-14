"""Redis-backed model state persistence with in-memory fallback.

Stores are keyed by ``{prefix}{store_name}:{token}`` in Redis.
Values are pickle-serialised (safe — only our own calibration dicts).

When Redis is unavailable the class degrades to a plain dict so the
application keeps working exactly as before.
"""

from __future__ import annotations

import logging
import pickle
from typing import Any, Iterator

from cortex.config import (
    PERSISTENCE_ENABLED,
    PERSISTENCE_KEY_PREFIX,
    PERSISTENCE_REDIS_URL,
)

logger = logging.getLogger(__name__)

_redis_client: Any | None = None
_redis_available: bool = False


async def init_persistence() -> None:
    """Connect to Redis for persistence. Safe to call even if Redis is down."""
    global _redis_client, _redis_available

    if not PERSISTENCE_ENABLED or not PERSISTENCE_REDIS_URL:
        logger.info("Model persistence disabled (PERSISTENCE_ENABLED=%s, URL=%s)",
                     PERSISTENCE_ENABLED, bool(PERSISTENCE_REDIS_URL))
        return

    try:
        import redis.asyncio as aioredis
        _redis_client = aioredis.from_url(
            PERSISTENCE_REDIS_URL,
            decode_responses=False,
            socket_connect_timeout=5,
        )
        await _redis_client.ping()
        _redis_available = True
        logger.info("Model persistence: Redis connected (%s)",
                     PERSISTENCE_REDIS_URL.split("@")[-1] if "@" in PERSISTENCE_REDIS_URL else PERSISTENCE_REDIS_URL)
    except Exception:
        _redis_available = False
        _redis_client = None
        logger.warning("Model persistence: Redis unavailable, using in-memory only", exc_info=True)


async def close_persistence() -> None:
    global _redis_client, _redis_available
    if _redis_client is not None:
        try:
            await _redis_client.aclose()
        except Exception:
            pass
    _redis_client = None
    _redis_available = False


class PersistentStore:
    """Dict-like store that lazily persists to Redis on write.

    Read operations always hit the in-memory dict (fast path).
    Write operations update both in-memory and Redis (if available).
    On startup, ``restore()`` loads all keys from Redis into memory.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._data: dict[str, dict] = {}

    def _redis_key(self, token: str) -> str:
        return f"{PERSISTENCE_KEY_PREFIX}{self._name}:{token}"

    # ── dict interface ──

    def __contains__(self, token: str) -> bool:
        return token in self._data

    def __getitem__(self, token: str) -> dict:
        return self._data[token]

    def __setitem__(self, token: str, value: dict) -> None:
        self._data[token] = value
        self._persist(token, value)

    def __delitem__(self, token: str) -> None:
        self._data.pop(token, None)
        self._delete(token)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def get(self, token: str, default: Any = None) -> Any:
        return self._data.get(token, default)

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def pop(self, token: str, *args) -> Any:
        val = self._data.pop(token, *args)
        self._delete(token)
        return val

    # ── persistence helpers (fire-and-forget via sync wrapper) ──
    # NOTE: pickle is safe here — we only serialise our own calibration dicts
    # containing numpy arrays and pandas objects. No external/untrusted data.

    def _persist(self, token: str, value: dict) -> None:
        if not _redis_available or _redis_client is None:
            return
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            loop.create_task(self._async_persist(token, value))
        except RuntimeError:
            pass

    async def _async_persist(self, token: str, value: dict) -> None:
        try:
            data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            await _redis_client.set(self._redis_key(token), data)
        except Exception:
            logger.warning("Failed to persist %s:%s to Redis", self._name, token, exc_info=True)

    def _delete(self, token: str) -> None:
        if not _redis_available or _redis_client is None:
            return
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            loop.create_task(self._async_delete(token))
        except RuntimeError:
            pass

    async def _async_delete(self, token: str) -> None:
        try:
            await _redis_client.delete(self._redis_key(token))
        except Exception:
            logger.warning("Failed to delete %s:%s from Redis", self._name, token, exc_info=True)

    # ── bulk operations ──

    async def restore(self) -> int:
        """Load all keys for this store from Redis into memory. Returns count."""
        if not _redis_available or _redis_client is None:
            return 0

        pattern = f"{PERSISTENCE_KEY_PREFIX}{self._name}:*"
        prefix_len = len(f"{PERSISTENCE_KEY_PREFIX}{self._name}:")
        count = 0
        try:
            async for key in _redis_client.scan_iter(match=pattern, count=100):
                try:
                    raw = await _redis_client.get(key)
                    if raw is None:
                        continue
                    token = key.decode()[prefix_len:] if isinstance(key, bytes) else key[prefix_len:]
                    self._data[token] = pickle.loads(raw)  # noqa: S301 — trusted internal data only
                    count += 1
                except Exception:
                    logger.warning("Failed to restore key %s", key, exc_info=True)
        except Exception:
            logger.warning("Failed to scan Redis for %s", self._name, exc_info=True)
        return count

    async def persist_all(self) -> int:
        """Write all in-memory data to Redis. Returns count."""
        if not _redis_available or _redis_client is None:
            return 0

        count = 0
        for token, value in self._data.items():
            try:
                data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                await _redis_client.set(self._redis_key(token), data)
                count += 1
            except Exception:
                logger.warning("Failed to persist %s:%s", self._name, token, exc_info=True)
        return count
