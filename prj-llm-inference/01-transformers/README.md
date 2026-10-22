# 01-transformers

Run the model **inside your Python process** with Hugging Face `transformers`.

```bash
python -m venv venv && venv\Scripts\pip install -r requirements.txt
python chat.py "Explain KV cache in two sentences"
python chat.py --stream "Write a haiku about GPUs"
```

- Default model `Qwen/Qwen2.5-0.5B-Instruct` (downloads to the HF cache on first run).
- Streaming uses `TextIteratorStreamer` + a generation thread.
- Pros: full control, any HF model. Cons: no batching / paged KV cache, one
  request at a time, slower than vLLM.

## Use cases
Research and experiments, custom generation settings, fine-tuned or niche Hub
models, and agent prototypes where you want the model in the same process as
your tools (finance agent, document summariser, RAG). See the
[parent README](../README.md) for the full comparison.
