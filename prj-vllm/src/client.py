"""Minimal async client for a vLLM OpenAI-compatible server.

Uses `httpx` directly (no `openai` SDK) — the same shape used in production:
POST /chat/completions, and for streaming, parse the SSE `data: ` lines and pull
`choices[0].delta.content`.
"""
from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from . import config

Message = dict[str, str]  # {"role": "system"|"user"|"assistant", "content": str}


def _headers() -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if config.VLLM_API_KEY:
        h["Authorization"] = f"Bearer {config.VLLM_API_KEY}"
    return h


def _url() -> str:
    return f"{config.VLLM_BASE_URL.rstrip('/')}/chat/completions"


async def chat(
    messages: list[Message],
    *,
    max_tokens: int = 256,
    temperature: float = 0.7,
    model: str | None = None,
    timeout: float | None = None,
) -> str:
    """One-shot, non-streaming completion. Returns the assistant's text."""
    payload = {
        "model": model or config.VLLM_SERVED_MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=timeout or config.REQUEST_TIMEOUT) as client:
        r = await client.post(_url(), json=payload, headers=_headers())
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


async def chat_stream(
    messages: list[Message],
    *,
    max_tokens: int = 256,
    temperature: float = 0.7,
    model: str | None = None,
    timeout: float | None = None,
) -> AsyncIterator[str]:
    """Streaming completion. Yields token strings (delta content) as they arrive."""
    payload = {
        "model": model or config.VLLM_SERVED_MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=timeout or config.REQUEST_TIMEOUT) as client:
        async with client.stream("POST", _url(), json=payload, headers=_headers()) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                    token = obj["choices"][0]["delta"].get("content", "")
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if token:
                    yield token
