"""Debate Transcript Persistence & Tiered Storage.

Stores adversarial debate results in a three-tier architecture:
  HOT   – In-memory ring buffer (last N debates, <5ms reads)
  WARM  – On-disk JSON-lines log (recent, queryable, append-only)
  COLD  – Compressed gzip archives (rotated daily, space-efficient)

Lifecycle:
  1. Every debate result is immediately written to HOT + WARM.
  2. A background rotation moves WARM entries older than WARM_RETENTION_HOURS
     into COLD archives grouped by date.
  3. Query methods transparently merge across tiers.

Thread-safety: All writes are serialized through a threading.Lock.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import shutil
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Configuration (env-backed) ──────────────────────────────────────────────

DATA_DIR = os.environ.get("DEBATE_STORE_DIR", "./data/debates")
HOT_CAPACITY = int(os.environ.get("DEBATE_HOT_CAPACITY", "200"))
WARM_RETENTION_HOURS = float(os.environ.get("DEBATE_WARM_RETENTION_HOURS", "72"))  # 3 days
COLD_COMPRESSION_LEVEL = int(os.environ.get("DEBATE_COLD_COMPRESSION", "6"))  # gzip 1-9
ROTATION_INTERVAL_SECONDS = float(os.environ.get("DEBATE_ROTATION_INTERVAL", "3600"))


# ── Transcript Schema ───────────────────────────────────────────────────────

def _make_transcript(
    debate_result: dict[str, Any],
    *,
    token: str = "",
    trade_size_usd: float = 0.0,
    direction: str = "",
) -> dict[str, Any]:
    """Normalize a run_debate() result into a storable transcript."""
    now = datetime.now(timezone.utc)
    return {
        "id": f"debate_{int(now.timestamp() * 1000)}_{token}",
        "timestamp": now.isoformat(),
        "epoch_ms": int(now.timestamp() * 1000),
        # Trade context
        "token": token,
        "direction": direction,
        "trade_size_usd": trade_size_usd,
        "strategy": debate_result.get("strategy", "spot"),
        # Decision
        "final_decision": debate_result.get("final_decision"),
        "final_confidence": debate_result.get("final_confidence"),
        "approval_score": debate_result.get("approval_score"),
        "recommended_size_pct": debate_result.get("recommended_size_pct"),
        "decision_changed": debate_result.get("decision_changed", False),
        "original_approved": debate_result.get("original_approved"),
        # Rounds (full transcript)
        "num_rounds": debate_result.get("num_rounds", 0),
        "rounds": debate_result.get("rounds", []),
        # Evidence
        "evidence_summary": debate_result.get("evidence_summary", {}),
        # Performance
        "elapsed_ms": debate_result.get("elapsed_ms", 0),
    }


# ── Tiered Storage Engine ───────────────────────────────────────────────────

class DebateStore:
    """Three-tier debate transcript store: HOT → WARM → COLD."""

    def __init__(
        self,
        data_dir: str = DATA_DIR,
        hot_capacity: int = HOT_CAPACITY,
        warm_retention_hours: float = WARM_RETENTION_HOURS,
    ):
        self._data_dir = Path(data_dir)
        self._warm_dir = self._data_dir / "warm"
        self._cold_dir = self._data_dir / "cold"
        self._hot_capacity = hot_capacity
        self._warm_retention_hours = warm_retention_hours

        # HOT tier: in-memory ring buffer
        self._hot: deque[dict[str, Any]] = deque(maxlen=hot_capacity)

        # WARM tier: append-only JSONL file
        self._warm_file = self._warm_dir / "transcripts.jsonl"

        # Thread safety
        self._lock = threading.Lock()

        # Background rotation state
        self._last_rotation = 0.0
        self._rotation_timer: threading.Timer | None = None

        # Ensure directories
        self._warm_dir.mkdir(parents=True, exist_ok=True)
        self._cold_dir.mkdir(parents=True, exist_ok=True)

        # Load existing warm data into hot cache
        self._hydrate_hot_from_warm()

        logger.info(
            "DebateStore initialized",
            extra={
                "data_dir": str(self._data_dir),
                "hot_capacity": hot_capacity,
                "warm_retention_hours": warm_retention_hours,
                "hot_loaded": len(self._hot),
            },
        )

    # ── Write Path ──────────────────────────────────────────────────────────

    def store(
        self,
        debate_result: dict[str, Any],
        *,
        token: str = "",
        trade_size_usd: float = 0.0,
        direction: str = "",
    ) -> dict[str, Any]:
        """Persist a debate transcript to HOT + WARM tiers.

        Returns the normalized transcript dict (with generated id).
        """
        transcript = _make_transcript(
            debate_result,
            token=token,
            trade_size_usd=trade_size_usd,
            direction=direction,
        )

        with self._lock:
            # HOT: push into ring buffer
            self._hot.append(transcript)

            # WARM: append to JSONL
            self._append_warm(transcript)

        logger.info(
            "Debate transcript stored",
            extra={
                "id": transcript["id"],
                "strategy": transcript["strategy"],
                "decision": transcript["final_decision"],
                "confidence": transcript["final_confidence"],
                "rounds": transcript["num_rounds"],
            },
        )

        # Trigger rotation check (non-blocking)
        self._maybe_rotate()

        return transcript

    def _append_warm(self, transcript: dict[str, Any]) -> None:
        """Append a single transcript to the JSONL warm file."""
        try:
            with open(self._warm_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(transcript, default=str) + "\n")
        except OSError as exc:
            logger.error("Failed to write warm transcript", extra={"error": str(exc)})

    # ── Read Path ───────────────────────────────────────────────────────────

    def get_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most recent transcripts from HOT tier."""
        with self._lock:
            items = list(self._hot)
        return items[-limit:][::-1]  # newest first

    def get_by_strategy(self, strategy: str, limit: int = 50) -> list[dict[str, Any]]:
        """Query transcripts by strategy across HOT + WARM tiers."""
        results: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        # HOT first
        with self._lock:
            for t in reversed(self._hot):
                if t.get("strategy") == strategy:
                    tid = t.get("id", "")
                    if tid not in seen_ids:
                        results.append(t)
                        seen_ids.add(tid)
                        if len(results) >= limit:
                            return results

        # WARM if we need more (skip IDs already in HOT)
        remaining = limit - len(results)
        if remaining > 0:
            warm_results = self._scan_warm(
                lambda t: t.get("strategy") == strategy and t.get("id", "") not in seen_ids,
                limit=remaining,
            )
            results.extend(warm_results)

        return results

    def get_by_token(self, token: str, limit: int = 50) -> list[dict[str, Any]]:
        """Query transcripts by token across HOT + WARM tiers."""
        results: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        with self._lock:
            for t in reversed(self._hot):
                if t.get("token") == token:
                    tid = t.get("id", "")
                    if tid not in seen_ids:
                        results.append(t)
                        seen_ids.add(tid)
                        if len(results) >= limit:
                            return results

        remaining = limit - len(results)
        if remaining > 0:
            warm_results = self._scan_warm(
                lambda t: t.get("token") == token and t.get("id", "") not in seen_ids,
                limit=remaining,
            )
            results.extend(warm_results)

        return results

    def get_by_id(self, transcript_id: str) -> dict[str, Any] | None:
        """Look up a specific transcript by ID."""
        with self._lock:
            for t in reversed(self._hot):
                if t.get("id") == transcript_id:
                    return t

        # Scan warm
        results = self._scan_warm(lambda t: t.get("id") == transcript_id, limit=1)
        return results[0] if results else None

    def get_decision_stats(self, hours: float = 24.0) -> dict[str, Any]:
        """Compute aggregate decision statistics over a time window."""
        cutoff_ms = int((time.time() - hours * 3600) * 1000)
        transcripts: list[dict[str, Any]] = []

        with self._lock:
            for t in self._hot:
                if t.get("epoch_ms", 0) >= cutoff_ms:
                    transcripts.append(t)

        if not transcripts:
            return {
                "period_hours": hours,
                "total_debates": 0,
                "decisions": {},
                "avg_confidence": 0,
                "avg_rounds": 0,
                "avg_elapsed_ms": 0,
                "decision_changed_count": 0,
                "by_strategy": {},
            }

        decisions: dict[str, int] = {}
        by_strategy: dict[str, dict[str, int]] = {}
        total_confidence = 0.0
        total_rounds = 0
        total_elapsed = 0.0
        changed_count = 0

        for t in transcripts:
            dec = t.get("final_decision", "unknown")
            decisions[dec] = decisions.get(dec, 0) + 1

            strat = t.get("strategy", "unknown")
            if strat not in by_strategy:
                by_strategy[strat] = {}
            by_strategy[strat][dec] = by_strategy[strat].get(dec, 0) + 1

            total_confidence += t.get("final_confidence", 0)
            total_rounds += t.get("num_rounds", 0)
            total_elapsed += t.get("elapsed_ms", 0)
            if t.get("decision_changed"):
                changed_count += 1

        n = len(transcripts)
        return {
            "period_hours": hours,
            "total_debates": n,
            "decisions": decisions,
            "avg_confidence": round(total_confidence / n, 4) if n else 0,
            "avg_rounds": round(total_rounds / n, 2) if n else 0,
            "avg_elapsed_ms": round(total_elapsed / n, 1) if n else 0,
            "decision_changed_count": changed_count,
            "by_strategy": by_strategy,
        }

    # ── WARM Scanning ───────────────────────────────────────────────────────

    def _scan_warm(
        self,
        predicate: Any,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Scan the warm JSONL file in reverse for matching transcripts."""
        results: list[dict[str, Any]] = []
        if not self._warm_file.exists():
            return results

        try:
            with open(self._warm_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Reverse scan (newest first)
            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line)
                    if predicate(t):
                        results.append(t)
                        if len(results) >= limit:
                            break
                except json.JSONDecodeError:
                    continue
        except OSError as exc:
            logger.warning("Failed to scan warm file", extra={"error": str(exc)})

        return results

    # ── HOT Hydration ───────────────────────────────────────────────────────

    def _hydrate_hot_from_warm(self) -> None:
        """Load the most recent entries from WARM into HOT on startup."""
        if not self._warm_file.exists():
            return

        try:
            with open(self._warm_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Take last N lines for hot cache
            recent_lines = lines[-self._hot_capacity:]
            for line in recent_lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line)
                    self._hot.append(t)
                except json.JSONDecodeError:
                    continue
        except OSError as exc:
            logger.warning("Failed to hydrate hot cache", extra={"error": str(exc)})

    # ── Cold Rotation ───────────────────────────────────────────────────────

    def _maybe_rotate(self) -> None:
        """Trigger cold rotation if enough time has passed."""
        now = time.time()
        if now - self._last_rotation < ROTATION_INTERVAL_SECONDS:
            return

        self._last_rotation = now
        # Run rotation in background thread
        t = threading.Thread(target=self._rotate_to_cold, daemon=True)
        t.start()

    def _rotate_to_cold(self) -> None:
        """Move expired warm entries into date-grouped cold archives."""
        if not self._warm_file.exists():
            return

        cutoff_ms = int((time.time() - self._warm_retention_hours * 3600) * 1000)
        keep_lines: list[str] = []
        archive_by_date: dict[str, list[str]] = {}
        rotated_count = 0

        try:
            with self._lock:
                with open(self._warm_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            t = json.loads(line)
                            epoch_ms = t.get("epoch_ms", 0)
                            if epoch_ms < cutoff_ms:
                                # Route to cold archive by date
                                ts = t.get("timestamp", "")
                                date_key = ts[:10] if len(ts) >= 10 else "unknown"
                                archive_by_date.setdefault(date_key, []).append(line + "\n")
                                rotated_count += 1
                            else:
                                keep_lines.append(line + "\n")
                        except json.JSONDecodeError:
                            keep_lines.append(line + "\n")

                # Rewrite warm file with only recent entries
                tmp = self._warm_file.with_suffix(".tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    f.writelines(keep_lines)
                tmp.rename(self._warm_file)

            # Write cold archives (outside lock)
            for date_key, lines in archive_by_date.items():
                archive_path = self._cold_dir / f"debates_{date_key}.jsonl.gz"
                mode = "ab"  # append to existing archive
                with gzip.open(archive_path, mode, compresslevel=COLD_COMPRESSION_LEVEL) as gz:
                    for line in lines:
                        gz.write(line.encode("utf-8"))

            if rotated_count > 0:
                logger.info(
                    "Cold rotation complete",
                    extra={
                        "rotated": rotated_count,
                        "kept_warm": len(keep_lines),
                        "archives": list(archive_by_date.keys()),
                    },
                )

        except Exception as exc:
            logger.error("Cold rotation failed", extra={"error": str(exc)})

    def search_cold(
        self,
        predicate: Any,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Search cold archives with optional date range filter.

        Args:
            predicate: Callable(transcript_dict) -> bool
            date_from: ISO date string (YYYY-MM-DD), inclusive
            date_to: ISO date string (YYYY-MM-DD), inclusive
            limit: Max results
        """
        results: list[dict[str, Any]] = []

        archive_files = sorted(self._cold_dir.glob("debates_*.jsonl.gz"), reverse=True)

        for archive in archive_files:
            # Extract date from filename
            fname = archive.stem.replace(".jsonl", "")  # debates_YYYY-MM-DD
            date_str = fname.replace("debates_", "")

            if date_from and date_str < date_from:
                continue
            if date_to and date_str > date_to:
                continue

            try:
                with gzip.open(archive, "rt", encoding="utf-8") as gz:
                    for line in gz:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            t = json.loads(line)
                            if predicate(t):
                                results.append(t)
                                if len(results) >= limit:
                                    return results
                        except json.JSONDecodeError:
                            continue
            except OSError as exc:
                logger.warning(
                    "Failed to read cold archive",
                    extra={"archive": str(archive), "error": str(exc)},
                )

        return results

    # ── Metrics ─────────────────────────────────────────────────────────────

    def get_storage_stats(self) -> dict[str, Any]:
        """Return storage statistics across all tiers."""
        warm_size = self._warm_file.stat().st_size if self._warm_file.exists() else 0
        cold_files = list(self._cold_dir.glob("debates_*.jsonl.gz"))
        cold_size = sum(f.stat().st_size for f in cold_files)

        return {
            "hot_count": len(self._hot),
            "hot_capacity": self._hot_capacity,
            "warm_file_bytes": warm_size,
            "warm_retention_hours": self._warm_retention_hours,
            "cold_archive_count": len(cold_files),
            "cold_total_bytes": cold_size,
            "cold_archives": [f.name for f in sorted(cold_files, reverse=True)[:10]],
        }

    def force_rotation(self) -> dict[str, Any]:
        """Manually trigger cold rotation. Returns rotation stats."""
        self._last_rotation = 0  # Reset timer to allow immediate rotation
        self._rotate_to_cold()
        return self.get_storage_stats()


# ── Module-level Singleton ──────────────────────────────────────────────────

_instance: DebateStore | None = None
_instance_lock = threading.Lock()


def get_debate_store() -> DebateStore:
    """Return the global DebateStore singleton."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = DebateStore()
    return _instance


def reset_debate_store() -> None:
    """Reset the singleton (for testing)."""
    global _instance
    _instance = None
