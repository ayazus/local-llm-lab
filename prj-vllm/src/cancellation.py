"""Barge-in: cancel an in-flight generation from another task.

Usage:
    barge = BargeIn()
    tokens = until_cancelled(client.chat_stream(msgs), barge)
    # ... from elsewhere (e.g. the user starts speaking):
    barge.trigger()
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator


class BargeIn:
    """A shared cancel signal. `.trigger()` from anywhere; loops check `.is_set()`."""

    def __init__(self) -> None:
        self._event = asyncio.Event()

    def trigger(self) -> None:
        self._event.set()

    def reset(self) -> None:
        self._event = asyncio.Event()

    def is_set(self) -> bool:
        return self._event.is_set()

    @property
    def event(self) -> asyncio.Event:
        return self._event


async def until_cancelled(
    source: AsyncIterator[str], barge: BargeIn
) -> AsyncIterator[str]:
    """Yield tokens from `source` until `barge` is triggered, then stop."""
    async for token in source:
        if barge.is_set():
            break
        yield token
