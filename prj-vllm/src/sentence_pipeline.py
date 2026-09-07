"""Double-streaming: LLM tokens -> sentence buffer -> queue -> downstream consumer.

Why: if you `await` a slow downstream step (TTS, another model, a network hop)
inline in the token loop, token generation for the *next* sentence is blocked for
that whole round trip. Handing finished sentences to a queue lets generation of
sentence N+1 run concurrently with the consumer handling sentence N, while the
consumer still receives sentences strictly in order (FIFO).

Engine-agnostic: `token_source` is any async iterator of str; `consume` is any
async callable taking one sentence.
"""
from __future__ import annotations

import asyncio
import re
from typing import AsyncIterator, Awaitable, Callable

# sentence-ending punctuation (incl. CJK), optional closing quote/bracket, then space
_SENTENCE_END = re.compile(r"[.!?。！？]['\"’”)\]]?(\s|$)")

# safety valve: force-flush an unpunctuated run-on once it gets this long
RUNON_WORDS = 40


def split_complete(buf: str) -> tuple[list[str], str]:
    """Split `buf` into (complete sentences, trailing remainder)."""
    out: list[str] = []
    while True:
        m = _SENTENCE_END.search(buf)
        if not m:
            break
        cut = m.end()
        piece = buf[:cut].strip()
        if piece:
            out.append(piece)
        buf = buf[cut:]
    return out, buf


async def stream_sentences(
    token_source: AsyncIterator[str],
    consume: Callable[[str], Awaitable[None]],
    *,
    stop: asyncio.Event | None = None,
) -> str:
    """Consume tokens, emit whole sentences to `consume` in order, pipelined.

    Returns the full assembled text. If `stop` is set mid-stream, generation and
    delivery halt and whatever was assembled so far is returned.
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    full: list[str] = []

    async def _sender() -> None:
        while True:
            item = await queue.get()
            if item is None:
                return
            if stop is not None and stop.is_set():
                continue  # drain silently — already cancelled
            await consume(item)

    sender = asyncio.create_task(_sender())
    buf = ""
    try:
        async for token in token_source:
            if stop is not None and stop.is_set():
                break
            buf += token
            sentences, buf = split_complete(buf)
            for s in sentences:
                full.append(s)
                await queue.put(s)
            if len(buf.split()) >= RUNON_WORDS:
                run = buf.strip()
                full.append(run)
                await queue.put(run)
                buf = ""
        tail = buf.strip()
        if tail and (stop is None or not stop.is_set()):
            full.append(tail)
            await queue.put(tail)
    finally:
        await queue.put(None)
        await sender
    return " ".join(full)
