"""Non-streaming completion against the vLLM server.

    python examples/01_oneshot.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.client import chat  # noqa: E402


async def main() -> None:
    reply = await chat(
        [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Name three uses for vLLM, one line each."},
        ],
        max_tokens=120,
    )
    print(reply)


if __name__ == "__main__":
    asyncio.run(main())
