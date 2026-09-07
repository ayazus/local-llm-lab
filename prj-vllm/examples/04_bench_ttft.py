"""TTFT + aggregate throughput across concurrency levels.

    python examples/04_bench_ttft.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.client import chat_stream  # noqa: E402
from src.metrics import StreamStats, instrument  # noqa: E402

PROMPT = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "Write a short paragraph about local LLM inference."},
]


async def one_call() -> StreamStats:
    stats = StreamStats()
    async for _ in instrument(chat_stream(PROMPT, max_tokens=128), stats):
        pass
    return stats


async def run_level(concurrency: int) -> None:
    results = await asyncio.gather(*(one_call() for _ in range(concurrency)))
    ttfts = sorted(s.ttft_s for s in results if s.ttft_s is not None)
    p50_ms = ttfts[len(ttfts) // 2] * 1000 if ttfts else 0.0
    agg_tps = sum(s.tokens_per_s for s in results)
    print(f"concurrency={concurrency:2d}  TTFT p50={p50_ms:7.0f} ms  aggregate {agg_tps:7.1f} tok/s")


async def main() -> None:
    for c in (1, 2, 4, 8):
        await run_level(c)


if __name__ == "__main__":
    asyncio.run(main())
