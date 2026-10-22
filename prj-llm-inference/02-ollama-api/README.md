# 02-ollama-api

Talk to a model served by **Ollama** over its local HTTP API (`localhost:11434`).
No Python dependencies (standard library only).

```bash
ollama pull qwen2.5:0.5b
python chat.py "Explain KV cache in two sentences"
python chat.py --stream "Write a haiku about GPUs"
```

- Env: `OLLAMA_HOST` (default `http://localhost:11434`), `OLLAMA_MODEL`.
- `/api/chat` with `stream:false` returns one JSON; with `stream:true` it returns
  newline-delimited JSON chunks ending with `"done": true`.
- Pros: easiest setup, model download/management built in. Cons: less tuning
  than vLLM, lower throughput.

## Use cases
Fastest way to give an app or **AI agent** a local chat endpoint: private finance
assistants, RAG over your documents, offline chatbots, and quick model swaps
(`ollama pull <model>`). See the [parent README](../README.md).
