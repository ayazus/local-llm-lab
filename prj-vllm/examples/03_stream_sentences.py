"""Stream tokens -> sentences -> a downstream consumer, pipelined, with barge-in.

The fake consumer sleeps ~0.8 s per sentence. Watch the log: generation of the
next sentence continues *while* the previous one is being "spoken".

    python examples/03_stream_sentences.py
"""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cancellation import BargeIn, until_cancelled  # noqa: E402
from src.client import chat_stream  # noqa: E402
from src.sentence_pipeline import stream_sentences  # noqa: E402

_t0 = time.perf_counter()


def _clock() -> str:
    return f"{time.perf_counter() - _t0:6.2f}s"


async def fake_tts(sentence: str) -> None:
    """Stand-in for a slow downstream step (TTS, another model, a network hop)."""
    print(f"{_clock()}  [speak start] {sentence!r}")
    await asyncio.sleep(0.8)
    print(f"{_clock()}  [speak done ] {sentence!r}")


async def main() -> None:
    barge = BargeIn()

    tokens = until_cancelled(
        chat_stream(
            [
                {"role": "system", "content": "You are a friendly interviewer."},
                {"role": "user", "content": "Give a short two-sentence welcome, then ask about my background."},
            ],
            max_tokens=200,
        ),
        barge,
    )

    async def _barge_after(delay: float) -> None:
        await asyncio.sleep(delay)
        # barge.trigger()   # uncomment to watch cancellation cut it short

    asyncio.create_task(_barge_after(2.0))

    full = await stream_sentences(tokens, fake_tts, stop=barge.event)
    print(f"\n{_clock()}  FULL: {full}")


if __name__ == "__main__":
    asyncio.run(main())
