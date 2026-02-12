"""In-memory async task store for background calibration jobs."""

import asyncio
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


_task_store: dict[str, dict[str, Any]] = {}


def create_task(endpoint: str) -> str:
    task_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    _task_store[task_id] = {
        "task_id": task_id,
        "status": TaskStatus.PENDING,
        "endpoint": endpoint,
        "result": None,
        "error": None,
        "created_at": now,
        "started_at": None,
        "completed_at": None,
    }
    return task_id


def get_task(task_id: str) -> dict[str, Any] | None:
    return _task_store.get(task_id)


async def run_in_background(task_id: str, sync_fn, *args, **kwargs) -> None:
    task = _task_store[task_id]
    task["status"] = TaskStatus.RUNNING
    task["started_at"] = datetime.now(timezone.utc).isoformat()
    try:
        result = await asyncio.to_thread(sync_fn, *args, **kwargs)
        task["status"] = TaskStatus.COMPLETED
        task["result"] = result
    except Exception as exc:
        task["status"] = TaskStatus.FAILED
        task["error"] = str(exc)
    finally:
        task["completed_at"] = datetime.now(timezone.utc).isoformat()

