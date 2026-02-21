"""Social feed sentiment aggregation — Twitter, Discord, Telegram.

Fetches social signals via HTTP APIs (no SDK dependencies) and scores
sentiment using the same Bayesian framework as ``cortex.news``.
"""

import logging
import os
import time
from typing import Any

import httpx

from cortex.config import SOCIAL_CACHE_TTL, TWITTER_BEARER_TOKEN

REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "")
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
DISCORD_CHANNEL_IDS = [c.strip() for c in os.environ.get("DISCORD_CHANNEL_IDS", "").split(",") if c.strip()]
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_IDS = [c.strip() for c in os.environ.get("TELEGRAM_CHAT_IDS", "").split(",") if c.strip()]

logger = logging.getLogger(__name__)

_social_cache: dict[str, Any] = {}
_cache_ts: float = 0.0

_SENTIMENT_KEYWORDS = {
    "bullish": ["bullish", "moon", "pump", "breakout", "ath", "buy", "long", "accumulate"],
    "bearish": ["bearish", "dump", "crash", "sell", "short", "rug", "scam", "liquidat"],
}


def _score_text(text: str) -> float:
    """Simple keyword sentiment scorer: returns -1.0 to 1.0."""
    lower = text.lower()
    bull = sum(1 for kw in _SENTIMENT_KEYWORDS["bullish"] if kw in lower)
    bear = sum(1 for kw in _SENTIMENT_KEYWORDS["bearish"] if kw in lower)
    total = bull + bear
    if total == 0:
        return 0.0
    return (bull - bear) / total


def fetch_twitter_sentiment(query: str = "solana", max_results: int = 20) -> dict[str, Any]:
    """Fetch recent tweets and compute aggregate sentiment."""
    if not TWITTER_BEARER_TOKEN:
        logger.debug("TWITTER_BEARER_TOKEN not set — returning neutral")
        return {"source": "twitter", "sentiment": 0.0, "count": 0, "items": []}

    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"}
    params = {"query": f"{query} -is:retweet lang:en", "max_results": min(max_results, 100)}

    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        logger.warning("Twitter API request failed")
        return {"source": "twitter", "sentiment": 0.0, "count": 0, "items": []}

    tweets = data.get("data", [])
    scores = [_score_text(t.get("text", "")) for t in tweets]
    avg = sum(scores) / len(scores) if scores else 0.0

    return {
        "source": "twitter",
        "sentiment": round(avg, 4),
        "count": len(tweets),
        "items": [
            {"text": t.get("text", "")[:200], "score": round(s, 4)}
            for t, s in zip(tweets, scores)
        ],
    }


def fetch_reddit_sentiment(query: str = "solana", subreddits: list[str] | None = None, limit: int = 25) -> dict[str, Any]:
    """Fetch recent Reddit posts from crypto subreddits and compute sentiment."""
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        logger.debug("Reddit credentials not set — returning neutral")
        return {"source": "reddit", "sentiment": 0.0, "count": 0, "items": []}

    subs = subreddits or ["solana", "cryptocurrency", "defi"]
    try:
        # OAuth2 app-only auth
        auth_resp = httpx.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET),
            data={"grant_type": "client_credentials"},
            headers={"User-Agent": "CortexBot/1.0"},
            timeout=10,
        )
        auth_resp.raise_for_status()
        token = auth_resp.json().get("access_token", "")
        if not token:
            return {"source": "reddit", "sentiment": 0.0, "count": 0, "items": []}

        headers = {"Authorization": f"Bearer {token}", "User-Agent": "CortexBot/1.0"}
        all_posts: list[dict] = []
        with httpx.Client(timeout=15) as client:
            for sub in subs:
                try:
                    resp = client.get(
                        f"https://oauth.reddit.com/r/{sub}/search",
                        headers=headers,
                        params={"q": query, "sort": "new", "limit": limit, "t": "day", "restrict_sr": "on"},
                    )
                    if resp.status_code == 200:
                        posts = resp.json().get("data", {}).get("children", [])
                        for p in posts:
                            d = p.get("data", {})
                            text = f"{d.get('title', '')} {d.get('selftext', '')[:300]}"
                            all_posts.append({"text": text, "sub": sub, "ups": d.get("ups", 0)})
                except Exception:
                    continue

        scores = [_score_text(p["text"]) for p in all_posts]
        avg = sum(scores) / len(scores) if scores else 0.0
        return {
            "source": "reddit",
            "sentiment": round(avg, 4),
            "count": len(all_posts),
            "items": [
                {"text": p["text"][:200], "score": round(s, 4), "sub": p["sub"]}
                for p, s in zip(all_posts[:20], scores[:20])
            ],
        }
    except Exception:
        logger.warning("Reddit API request failed", exc_info=True)
        return {"source": "reddit", "sentiment": 0.0, "count": 0, "items": []}


def fetch_discord_sentiment(keyword: str = "solana", limit: int = 50) -> dict[str, Any]:
    """Fetch recent messages from configured Discord channels and compute sentiment."""
    if not DISCORD_BOT_TOKEN or not DISCORD_CHANNEL_IDS:
        logger.debug("Discord credentials not set — returning neutral")
        return {"source": "discord", "sentiment": 0.0, "count": 0, "items": []}

    try:
        headers = {"Authorization": f"Bot {DISCORD_BOT_TOKEN}"}
        messages: list[dict] = []
        kw_lower = keyword.lower()

        with httpx.Client(timeout=15) as client:
            for ch_id in DISCORD_CHANNEL_IDS[:5]:
                try:
                    resp = client.get(
                        f"https://discord.com/api/v10/channels/{ch_id}/messages",
                        headers=headers,
                        params={"limit": limit},
                    )
                    if resp.status_code == 200:
                        for msg in resp.json():
                            content = msg.get("content", "")
                            if kw_lower in content.lower():
                                messages.append({"text": content, "channel": ch_id})
                except Exception:
                    continue

        scores = [_score_text(m["text"]) for m in messages]
        avg = sum(scores) / len(scores) if scores else 0.0
        return {
            "source": "discord",
            "sentiment": round(avg, 4),
            "count": len(messages),
            "items": [
                {"text": m["text"][:200], "score": round(s, 4)}
                for m, s in zip(messages[:20], scores[:20])
            ],
        }
    except Exception:
        logger.warning("Discord API request failed", exc_info=True)
        return {"source": "discord", "sentiment": 0.0, "count": 0, "items": []}


def fetch_telegram_sentiment(keyword: str = "solana", limit: int = 50) -> dict[str, Any]:
    """Fetch recent messages from configured Telegram chats and compute sentiment."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_IDS:
        logger.debug("Telegram credentials not set — returning neutral")
        return {"source": "telegram", "sentiment": 0.0, "count": 0, "items": []}

    try:
        base_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
        messages: list[dict] = []
        kw_lower = keyword.lower()

        with httpx.Client(timeout=15) as client:
            # getUpdates to fetch recent messages
            resp = client.get(f"{base_url}/getUpdates", params={"limit": 100, "timeout": 0})
            if resp.status_code == 200:
                updates = resp.json().get("result", [])
                for upd in updates:
                    msg = upd.get("message", {})
                    text = msg.get("text", "")
                    chat_id = str(msg.get("chat", {}).get("id", ""))
                    if chat_id in TELEGRAM_CHAT_IDS and kw_lower in text.lower():
                        messages.append({"text": text, "chat": chat_id})

        scores = [_score_text(m["text"]) for m in messages]
        avg = sum(scores) / len(scores) if scores else 0.0
        return {
            "source": "telegram",
            "sentiment": round(avg, 4),
            "count": len(messages),
            "items": [
                {"text": m["text"][:200], "score": round(s, 4)}
                for m, s in zip(messages[:20], scores[:20])
            ],
        }
    except Exception:
        logger.warning("Telegram API request failed", exc_info=True)
        return {"source": "telegram", "sentiment": 0.0, "count": 0, "items": []}


def fetch_social_sentiment(token: str = "solana") -> dict[str, Any]:
    """Aggregate sentiment across all available social sources with caching."""
    global _social_cache, _cache_ts

    now = time.time()
    cache_key = token.lower()
    if cache_key in _social_cache and (now - _cache_ts) < SOCIAL_CACHE_TTL:
        return _social_cache[cache_key]

    twitter = fetch_twitter_sentiment(query=token)
    reddit = fetch_reddit_sentiment(query=token)
    discord = fetch_discord_sentiment(keyword=token)
    telegram = fetch_telegram_sentiment(keyword=token)

    sources = [twitter, reddit, discord, telegram]
    sentiments = [s["sentiment"] for s in sources if s["count"] > 0]
    overall = sum(sentiments) / len(sentiments) if sentiments else 0.0

    result = {
        "token": token,
        "overall_sentiment": round(overall, 4),
        "source_count": len([s for s in sources if s["count"] > 0]),
        "total_items": sum(s["count"] for s in sources),
        "sources": sources,
        "timestamp": now,
    }

    _social_cache[cache_key] = result
    _cache_ts = now
    return result

