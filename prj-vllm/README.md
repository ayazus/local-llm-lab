# prj-vllm

Serving an LLM locally with **vLLM** (OpenAI-compatible server) and consuming it
with **token streaming** — plus the pattern that makes streaming usable for
real-time voice/chat: **token → sentence → downstream**, pipelined, with barge-in.

Extracted from a production AI-interview system (Qwen2.5-14B-AWQ on a single GPU).

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

See [`docs/launch-flags.md`](docs/launch-flags.md) for what each flag trades off.

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
