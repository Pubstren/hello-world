"""Asynchronous JSONL output helpers."""

from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path
import json
from typing import Any


class JsonlWriter:
    """Single-consumer JSONL writer with bounded queues for realtime ingestion."""

    def __init__(self, output_dir: Path, queue_size: int = 100_000) -> None:
        if queue_size <= 0:
            raise ValueError("writer queue size must be positive")
        self.output_dir = output_dir
        self._queue: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue(
            maxsize=queue_size
        )
        self.dropped = Counter()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._task = asyncio.create_task(self._run(), name="jsonl-writer")

    async def close(self) -> None:
        if self._task is None:
            return
        await self._queue.put(None)
        await self._task

    def write_nowait(self, file_name: str, record: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait((file_name, record))
        except asyncio.QueueFull:
            self.dropped[file_name] += 1

    async def _run(self) -> None:
        handles: dict[str, Any] = {}
        try:
            while True:
                item = await self._queue.get()
                if item is None:
                    break

                file_name, record = item
                handle = handles.get(file_name)
                if handle is None:
                    handle = (self.output_dir / file_name).open(
                        "a", encoding="utf-8", buffering=1
                    )
                    handles[file_name] = handle
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        finally:
            for handle in handles.values():
                handle.close()
