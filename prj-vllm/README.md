# prj-vllm

Serving an LLM locally with **vLLM** (OpenAI-compatible server) and consuming it
with **token streaming** — plus the pattern that makes streaming usable for
real-time voice/chat: **token → sentence → downstream**, pipelined, with barge-in.

Patterns extracted from a production AI-interview system (Qwen2.5-14B-AWQ on a
single consumer GPU).

---

## What is vLLM?

**vLLM** is an open-source **inference engine** for large language models — the
software that actually runs a model and answers requests. It was created at UC
Berkeley (Sky Computing Lab), is Apache-2.0 licensed, and is now one of the most
widely used engines for self-hosted LLMs.

Two ways to use it:

1. **A Python library** — `from vllm import LLM`, for offline / batch generation.
2. **An OpenAI-compatible HTTP server** — `vllm.entrypoints.openai.api_server`,
   which exposes `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, with
   SSE streaming, tool calling and structured output. This folder uses the server.

### Purpose

Serve an LLM **efficiently to many concurrent requests on limited GPU hardware**,
behind an API your existing code already knows how to call. It is the piece
between "I have model weights on disk" and "my app can call a `/chat` endpoint".

### Why it's fast — the key ideas

| Technique | What it does | Why it matters |
|---|---|---|
| **PagedAttention** | Stores the attention KV cache in fixed-size **pages**, managed like OS virtual memory, instead of one big contiguous block per request. | Naive allocation wastes 60–80% of KV memory to fragmentation and over-reservation. Paging cuts that to a few percent, so **2–4× more requests fit in the same VRAM**. |
| **Continuous batching** | New requests join the in-flight batch the moment any sequence finishes, rather than waiting for the whole batch. | The GPU stays saturated. Under load, throughput is many times a naive "wait for the batch" loop. |
| **Optimized CUDA kernels** | Fused attention/MLP kernels, FlashAttention / FlashInfer, CUDA graphs. | Lower per-token cost and lower latency. |
| **Quantization** | Loads AWQ, GPTQ, FP8, INT8, GGUF checkpoints. | A 14B model in 4-bit AWQ is ~9 GB of weights instead of ~28 GB — it fits on an 8 GB card with room for the KV cache. |
| **Prefix caching** | Reuses KV for a shared prompt prefix across requests. | If every request starts with the same long system prompt, that prefix is encoded **once**. Big TTFT win for agents/interviewers. |
| **Speculative decoding, chunked prefill, tensor/pipeline parallelism** | Draft-model speedups, smoother prefill under load, multi-GPU splitting. | Scale up when one GPU or one model isn't enough. |

### Benefits in practice

- **OpenAI-compatible** — point the OpenAI SDK (or plain `httpx`) at
  `http://localhost:8000/v1` and most code works unchanged. Streaming, `tools`,
  and JSON-schema output are supported.
- **High throughput per GPU** — serve a small team from one card.
- **Predictable cost** — no per-token bill; the GPU is the only cost.
- **Privacy / offline** — prompts never leave the machine.
- **Production features included** — metrics endpoint, health checks, LoRA
  adapter hot-swap, guided decoding, request cancellation.

### Trade-offs / when *not* to use it

- **Linux + NVIDIA (CUDA)** is the supported path. AMD ROCm and CPU exist but lag;
  no native Windows — use **WSL2** or Docker.
- Heavier to install and operate than **Ollama** / **llama.cpp**, which are the
  better choice for single-user local use on a Mac/CPU or for GGUF simplicity.
- Batching optimizes **throughput**, not single-request latency — one lone request
  is not dramatically faster than plain `transformers`.
- Needs enough VRAM for weights **plus** KV cache; very long context or very high
  concurrency still hits a memory wall (see [`docs/launch-flags.md`](docs/launch-flags.md)).

### How it compares

| Engine | Sweet spot |
|---|---|
| **vLLM** | High-throughput GPU serving, many concurrent users, OpenAI API |
| TGI (Hugging Face) | Similar niche, tight HF ecosystem integration |
| TensorRT-LLM | Absolute max NVIDIA performance, more build effort |
| Ollama / llama.cpp | Single-user local, Mac/CPU, GGUF, one-command simplicity |
| Raw `transformers` | Research, custom generation loops, small scale |

---

## 1. Start the server

Needs Linux + an NVIDIA GPU and `pip install vllm`.

```bash
python download_model.py                 # -> models/qwen-14b-awq  (git-ignored)
bash start_vllm.sh                        # OpenAI server on :8000
```

`start_vllm.sh` runs:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model ./models/qwen-14b-awq \
    --served-model-name qwen-14b-awq \
    --quantization awq_marlin --dtype float16 \
    --max-model-len 24000 --gpu-memory-utilization 0.85 \
    --host 0.0.0.0 --port 8000
```

See [`docs/launch-flags.md`](docs/launch-flags.md) for what each flag trades off
and how to size VRAM.

## 2. Client (runs anywhere — no GPU)

```bash
python -m venv venv && venv\Scripts\pip install -r requirements.txt
cp .env.example .env      # adjust VLLM_BASE_URL / VLLM_SERVED_MODEL_NAME
```

```python
from src.client import chat, chat_stream

text = await chat([{"role": "user", "content": "hi"}])          # non-streaming

async for token in chat_stream([{"role": "user", "content": "hi"}]):
    print(token, end="")                                        # streaming
```

Plain `httpx` — no `openai` SDK. Streaming parses the SSE `data:` lines and pulls
`choices[0].delta.content`. See [`docs/streaming.md`](docs/streaming.md).

## 3. Examples

| File | Shows |
|---|---|
| [`examples/01_oneshot.py`](examples/01_oneshot.py) | one non-streaming completion |
| [`examples/02_stream_tokens.py`](examples/02_stream_tokens.py) | print tokens live + TTFT / tok/s |
| [`examples/03_stream_sentences.py`](examples/03_stream_sentences.py) | tokens → sentences → slow consumer, pipelined + barge-in |
| [`examples/04_bench_ttft.py`](examples/04_bench_ttft.py) | TTFT p50 + aggregate throughput at concurrency 1/2/4/8 |

Browser: open [`web/chat_test.html`](web/chat_test.html) — streams from the server
directly via `fetch` + `ReadableStream` (server must allow the page's origin;
vLLM's OpenAI server allows all origins by default).

## 4. The double-streaming pattern

`src/sentence_pipeline.py`: buffer tokens, flush on sentence-end punctuation, push
each sentence onto an `asyncio.Queue`; a separate sender task runs the slow
downstream step (TTS, another model). Generation of sentence *N+1* then overlaps
the consumer handling sentence *N*, and sentences still arrive in order.
`src/cancellation.py` adds barge-in — stop the token loop the moment the user
speaks. Full rationale in [`docs/double-streaming.md`](docs/double-streaming.md).

## Files

```
prj-vllm/
├── start_vllm.sh          server launch
├── download_model.py      fetch AWQ weights -> models/ (git-ignored)
├── src/
│   ├── config.py          VLLM_BASE_URL / served-model-name / timeout from env
│   ├── client.py          chat() + chat_stream()  (httpx, no openai SDK)
│   ├── sentence_pipeline.py  tokens -> sentences -> queue -> consumer
│   ├── cancellation.py    BargeIn + until_cancelled()
│   └── metrics.py         StreamStats: TTFT, gaps, tok/s
├── examples/              01..04
├── web/chat_test.html     browser SSE streaming chat
└── docs/                  launch-flags · streaming · double-streaming
```

## Notes

- `.env` and `models/` are git-ignored. `.env.example` holds **placeholders only**
  — there are no secrets in this folder.
- vLLM is Apache-2.0; the Qwen models keep their own licences.
