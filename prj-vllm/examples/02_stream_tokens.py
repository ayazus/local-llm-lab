"""Print tokens as they stream in, then show TTFT / throughput.

    python examples/02_stream_tokens.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.client import chat_stream  # noqa: E402
from src.metrics import StreamStats, instrument  # noqa: E402


async def main() -> None:
    stats = StreamStats()
    stream = instrument(
        chat_stream(
            [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "Explain streaming token generation in three sentences."},
            ],
            max_tokens=200,
        ),
        stats,
    )
    async for token in stream:
        print(token, end="", flush=True)
    print("\n---")
    print(stats.summary())


if __name__ == "__main__":
    asyncio.run(main())
