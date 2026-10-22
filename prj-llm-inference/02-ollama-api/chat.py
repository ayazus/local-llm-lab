"""Ollama HTTP API (stdlib only): one-shot and streaming chat.

    ollama pull qwen2.5:0.5b
    python chat.py "Explain KV cache in two sentences"
    python chat.py --stream "Write a haiku about GPUs"
"""
import argparse
import json
import os
import urllib.request

BASE = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")


def _post(path: str, payload: dict):
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=120)


def messages_for(prompt: str):
    return [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": prompt},
    ]


def chat(prompt: str, model: str = MODEL) -> str:
    payload = {"model": model, "messages": messages_for(prompt), "stream": False}
    with _post("/api/chat", payload) as r:
        return json.loads(r.read())["message"]["content"]


def chat_stream(prompt: str, model: str = MODEL):
    """Ollama streams newline-delimited JSON objects (not SSE)."""
    payload = {"model": model, "messages": messages_for(prompt), "stream": True}
    with _post("/api/chat", payload) as r:
        for line in r:
            if not line.strip():
                continue
            obj = json.loads(line)
            piece = obj.get("message", {}).get("content", "")
            if piece:
                yield piece
            if obj.get("done"):
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--stream", action="store_true")
    a = ap.parse_args()
    if a.stream:
        for piece in chat_stream(a.prompt, a.model):
            print(piece, end="", flush=True)
        print()
    else:
        print(chat(a.prompt, a.model))


if __name__ == "__main__":
    main()
