# local-llm-lab

A hands-on lab for running large language models **locally** with
[**vLLM**](https://github.com/vllm-project/vllm) — self-hosted, OpenAI-compatible
serving with real **token streaming**, on a single consumer GPU.

Each topic is a self-contained folder with runnable code and a short write-up of
what was learned. The goal is a practical reference for building on a local LLM:
how to serve it efficiently, stream from it, keep latency low, and wire it into a
real application.

> Research / learning repo — not a packaged library. vLLM and the models it runs
> keep their own upstream licences.

---

## Why local inference

Running the model yourself — instead of calling a hosted API — buys you:

- **Privacy** — prompts and data never leave the machine. Required for regulated
  domains (health, HR, legal, finance) and for anything with user PII.
- **Cost** — no per-token billing. Once the GPU is paid for, inference is free;
  heavy or bursty workloads stop being a line item.
- **Control** — pin an exact model version, tune sampling and context length, run
  offline, and never get rate-limited or deprecated out from under you.
- **Latency** — no network hop to a provider; time-to-first-token is bounded by
  your own hardware.

The cost is operational: you manage the GPU, the server, and the memory budget.
This repo is about doing that well.

## What vLLM brings

vLLM is an open-source inference engine (UC Berkeley, Apache-2.0) built for
**high-throughput, memory-efficient** serving:

- **PagedAttention** — the KV cache is managed in fixed-size pages like OS virtual
  memory, so almost no VRAM is wasted to fragmentation. Far more concurrent
  requests fit on the same card.
- **Continuous batching** — a new request joins the running batch the instant a
  slot frees, instead of waiting for the whole batch to finish. The GPU stays
  saturated.
- **Quantization** (AWQ / GPTQ / FP8 / GGUF) — run a 14B model in ~9 GB.
- **OpenAI-compatible server** — `/v1/chat/completions` etc., SSE streaming, tool
  calling, and guided/structured output. Existing OpenAI client code mostly just
  works against `http://localhost:8000/v1`.

Deep dive in [`prj-vllm/README.md`](prj-vllm/README.md).

---

## Folders

| Folder | What it covers |
|---|---|
| [`prj-vllm`](prj-vllm) | Serve a model with the vLLM OpenAI server; an `httpx` streaming + non-streaming client (no `openai` SDK); the **token → sentence → downstream** pipeline ("double-streaming") with **barge-in**; and **TTFT / tokens-per-second** metrics. Patterns extracted from a production real-time voice system. |

### Roadmap (planned folders)

| Folder | Idea |
|---|---|
| `prj-structured` | guided decoding — JSON-schema / regex / grammar constrained output |
| `prj-tools` | function / tool calling, parallel calls, a minimal agent loop |
| `prj-rag` | retrieval-augmented generation on a local vLLM backend (embeddings + vector store + streamed answers) |
| `prj-batch` | offline high-throughput batch inference with the in-process `LLM` class |
| `prj-bench` | latency + throughput benchmarks: TTFT, tok/s, concurrency sweeps, prefix-cache impact |

---

## Repository layout

```
local-llm-lab/
├── README.md                 this file — the overall project
├── LICENSE                   MIT (own code; vLLM / models keep their licences)
├── .gitignore                excludes venvs, model weights, .env, secrets
│
└── prj-vllm/
    ├── README.md             what vLLM is, why, and how this folder uses it
    ├── requirements.txt      client deps only (httpx, python-dotenv)
    ├── .env.example          VLLM_BASE_URL / VLLM_SERVED_MODEL_NAME / MODEL_PATH (placeholders only)
    ├── start_vllm.sh         the OpenAI-compatible server launch command
    ├── download_model.py     fetch the AWQ weights into models/ (git-ignored)
    ├── src/
    │   ├── config.py         connection settings from the environment
    │   ├── client.py         httpx client: chat() + chat_stream()  (no openai SDK)
    │   ├── sentence_pipeline.py  tokens → sentence buffer → asyncio.Queue → consumer
    │   ├── cancellation.py   barge-in: stop generation from another task
    │   └── metrics.py        TTFT, inter-token latency, tokens/sec
    ├── examples/             01 one-shot · 02 stream tokens · 03 stream sentences · 04 benchmark
    ├── web/chat_test.html    minimal browser SSE streaming chat page
    └── docs/                 launch flags · streaming wire format · double-streaming
```

Model weights, virtualenvs, generated output, and `.env` files stay local — the
`.gitignore` keeps them out. Only `*.env.example` (placeholders) is committed.

---

## Hardware context

Everything here targets a **single consumer GPU (~8 GB VRAM)**. A 14B model in
4-bit AWQ (~9 GB of weights) is the reference workload; smaller models leave more
room for a larger KV cache and higher concurrency. vLLM runs on Linux + NVIDIA
(CUDA); on Windows use WSL2 or a container. The `httpx` **client** side of each
folder runs anywhere, no GPU needed.

## Using a folder

Each `prj-*` folder is self-contained:

```bash
cd prj-vllm
python -m venv venv && venv\Scripts\pip install -r requirements.txt
cp .env.example .env          # point at your server
```

Then follow that folder's `README.md` — it has the exact `vllm serve` command,
client snippets, and how to run the examples.

---

## Licence

Original code and docs in this repository: [MIT](LICENSE). vLLM is Apache-2.0; the
Qwen (and any other) models keep their own upstream licences — check them before
non-personal use.
