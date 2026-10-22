"""llama.cpp via llama-cpp-python (GGUF): one-shot and streaming chat.

    pip install -r requirements.txt
    python chat.py --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf "Hi"
    python chat.py --model models/<file>.gguf --stream "Write a haiku"
"""
import argparse
import os

from llama_cpp import Llama

DEFAULT_MODEL = os.getenv("GGUF_PATH", "models/qwen2.5-0.5b-instruct-q4_k_m.gguf")


def load(path: str, n_gpu_layers: int = 0) -> Llama:
    # n_gpu_layers=-1 offloads everything if built with CUDA/Metal support
    return Llama(model_path=path, n_ctx=4096, n_gpu_layers=n_gpu_layers, verbose=False)


def messages_for(prompt: str):
    return [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": prompt},
    ]


def chat(llm: Llama, prompt: str, max_tokens: int = 200) -> str:
    out = llm.create_chat_completion(messages=messages_for(prompt), max_tokens=max_tokens)
    return out["choices"][0]["message"]["content"]


def chat_stream(llm: Llama, prompt: str, max_tokens: int = 200):
    for chunk in llm.create_chat_completion(
        messages=messages_for(prompt), max_tokens=max_tokens, stream=True
    ):
        piece = chunk["choices"][0]["delta"].get("content", "")
        if piece:
            yield piece


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--gpu-layers", type=int, default=0)
    ap.add_argument("--stream", action="store_true")
    a = ap.parse_args()
    if not os.path.exists(a.model):
        raise SystemExit(f"GGUF not found: {a.model}\nPut a .gguf file in models/ (git-ignored).")
    llm = load(a.model, a.gpu_layers)
    if a.stream:
        for piece in chat_stream(llm, a.prompt):
            print(piece, end="", flush=True)
        print()
    else:
        print(chat(llm, a.prompt))


if __name__ == "__main__":
    main()
