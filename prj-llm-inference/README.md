# prj-llm-inference — 3 Ways to Run and Talk to a Local LLM (Transformers, Ollama API, llama.cpp)

**Run a large language model on your own machine and call it from Python — three
different ways, with working one-shot and token-streaming examples for each.**

This folder compares the three most common approaches to **local LLM inference**
so you can pick the right one for your project, whether that is a chatbot, a
**finance agent**, an **agentic AI** workflow, a RAG pipeline, or an offline
assistant that never sends data to a cloud API.

> Keywords: local LLM, LLM inference, Hugging Face Transformers, Ollama API,
> llama.cpp, llama-cpp-python, GGUF, quantization, token streaming, streaming
> chat, offline AI, private AI, self-hosted LLM, Qwen2.5, agentic AI, AI agents,
> finance agent, RAG, Python.

---

## The three techniques at a glance

| Folder | Technique | How it runs | Setup effort | Hardware | Best for |
|---|---|---|---|---|---|
| [`01-transformers`](01-transformers) | **Hugging Face Transformers** | model loaded *inside* your Python process | medium (PyTorch + weights) | GPU recommended, CPU works for small models | research, fine-tuning, full control, any model on the Hub |
| [`02-ollama-api`](02-ollama-api) | **Ollama HTTP API** | separate local server on `localhost:11434` | **easiest** (`ollama pull`) | CPU or GPU, handled for you | quick prototypes, apps that just need a chat endpoint, easy model swapping |
| [`03-llama-cpp`](03-llama-cpp) | **llama.cpp** (`llama-cpp-python`, GGUF) | quantized single-file model, in-process | easy (one `.gguf` file) | **CPU / low VRAM / Mac** | small memory footprint, edge and laptop deployments |

Each folder has the same interface — `chat()` for a one-shot reply and
`chat_stream()` for **token-by-token streaming** — so you can swap the backend
without changing the rest of your app.

For **high-throughput serving to many users**, see the sibling folder
[`../prj-vllm`](../prj-vllm) (vLLM + OpenAI-compatible API + streaming pipeline).

---

## Why run an LLM locally?

- **Privacy** — prompts, documents and customer data never leave your machine.
  Important for finance, health, HR and legal data.
- **Cost** — no per-token API bill; run as many experiments as you want.
- **Offline** — works without internet once the model is downloaded.
- **Control** — pin an exact model version, tune sampling, no rate limits.
- **Low latency** — no network round-trip to a hosted provider.

---

## Which one should I use?

| If you need… | Use |
|---|---|
| The quickest path to "a local model answering my prompts" | **Ollama API** |
| To modify generation, use custom Hub models, fine-tune, or do research | **Transformers** |
| To run on a laptop / CPU / small GPU with minimal RAM | **llama.cpp (GGUF)** |
| To serve many concurrent users at high throughput | [**vLLM**](../prj-vllm) |
| Tool-calling agents against a local OpenAI-style endpoint | Ollama or vLLM (both expose chat APIs) |

---

## Use cases

These techniques are the building blocks. The model answers; your code adds the
tools and data around it.

### Finance agent (local, private)
Summarise earnings reports and filings, extract figures from statements into
JSON, classify transactions, explain a portfolio's risk in plain language, or
draft analyst notes — **with sensitive financial data kept on your own
hardware**. Pair a local LLM with Python tools (pandas, a market-data API, a
calculator) so the model decides *what to call* and your code does the exact
arithmetic.

### Agentic AI and AI agents
An agent is an LLM in a loop: *think → call a tool → read the result → repeat*.
Any of the three backends can drive that loop. Use streaming (`chat_stream`) to
show the agent's reasoning live, and a small, fast model for routing steps.
Typical tools: web search, file readers, SQL, code execution, email/calendar.

### RAG (chat with your documents)
Embed your PDFs/notes, retrieve the relevant chunks, and pass them to the local
model as context. Everything — documents, embeddings, answers — stays offline.

### Customer-support and internal assistants
Answer FAQs from company knowledge without sending tickets to a third party.

### Voice assistants and real-time apps
Streaming tokens let you start text-to-speech on the first sentence while the
rest is still generating (see the sentence-pipeline pattern in
[`../prj-vllm`](../prj-vllm)).

### Data extraction and automation
Turn messy text (invoices, emails, logs) into structured JSON, tag and route
documents, generate SQL from questions, or write first-draft reports.

### Learning and prototyping
Compare models and backends on your own laptop before committing to a hosted API
or a GPU server.

---

## Quick start

```bash
# Ollama (easiest)
ollama pull qwen2.5:0.5b
cd 02-ollama-api && python chat.py --stream "Explain compound interest simply"

# Transformers
cd 01-transformers && pip install -r requirements.txt
python chat.py "Summarise the risks of a 100% stock portfolio"

# llama.cpp (GGUF)
cd 03-llama-cpp && pip install -r requirements.txt
python chat.py --model models/<your-model>.gguf --stream "Hello"
```

Defaults use the small **Qwen2.5-0.5B-Instruct** model so everything runs on
modest hardware; swap in a larger model for better answers.

---

## Streaming, in one line each

- **Transformers** — `TextIteratorStreamer` yields text while `generate()` runs in a thread.
- **Ollama** — `/api/chat` with `stream: true` returns newline-delimited JSON chunks.
- **llama.cpp** — `create_chat_completion(stream=True)` yields delta chunks.

---

## Notes

- Model files and virtualenvs are git-ignored; nothing here needs an API key.
- Each backend keeps its own upstream licence (Transformers: Apache-2.0,
  Ollama: MIT, llama.cpp: MIT); models have their own licences — check before
  commercial use.
- Part of the [`local-llm-lab`](..) repository.
