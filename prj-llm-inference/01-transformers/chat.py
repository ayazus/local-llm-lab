"""Hugging Face transformers: one-shot and streaming chat, in-process.

    pip install -r requirements.txt
    python chat.py "Explain KV cache in two sentences"
    python chat.py --stream "Write a haiku about GPUs"
"""
import argparse
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # small; swap for any chat model


def load(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )
    return tok, model


def build_inputs(tok, model, prompt: str):
    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok([text], return_tensors="pt").to(model.device)


def chat(tok, model, prompt: str, max_new_tokens: int = 200) -> str:
    inputs = build_inputs(tok, model, prompt)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=True, temperature=0.7)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def chat_stream(tok, model, prompt: str, max_new_tokens: int = 200):
    """Yield text pieces as they are generated."""
    inputs = build_inputs(tok, model, prompt)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(**inputs, streamer=streamer, max_new_tokens=max_new_tokens,
                  do_sample=True, temperature=0.7)
    threading.Thread(target=model.generate, kwargs=kwargs, daemon=True).start()
    for piece in streamer:
        yield piece


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--stream", action="store_true")
    a = ap.parse_args()
    tok, model = load(a.model)
    if a.stream:
        for piece in chat_stream(tok, model, a.prompt):
            print(piece, end="", flush=True)
        print()
    else:
        print(chat(tok, model, a.prompt))


if __name__ == "__main__":
    main()
