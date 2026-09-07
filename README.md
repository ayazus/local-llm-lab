# local-llm-lab

A hands-on lab for running LLMs locally with **vLLM** — OpenAI-compatible
serving, **token streaming**, and the streaming patterns needed for real-time
voice/chat apps. Each topic lives in its own folder.

> Research / learning repo — not a library. vLLM and the models keep their own
> upstream licences.

---

## Folders

| Folder | What it covers |
|---|---|
| [`prj-vllm`](prj-vllm) | vLLM OpenAI server launch, an httpx streaming/non-streaming client, and the **token → sentence → downstream** pipeline ("double-streaming") with barge-in and TTFT metrics |

_More folders to come (structured output, tool calling, RAG on a local backend, batch inference, benchmarks)._

---

## Repository structure

```
local-llm-lab/
├── README.md
├── LICENSE                     MIT (own code; vLLM / models keep their licences)
├── .gitignore                  excludes venvs, model weights, .env, secrets
│
└── prj-vllm/
    ├── README.md
    ├── requirements.txt        client deps (httpx, python-dotenv); vLLM installed separately
    ├── .env.example            VLLM_BASE_URL / VLLM_SERVED_MODEL_NAME / MODEL_PATH (no secrets)
    ├── start_vllm.sh           the OpenAI-compatible server launch command
    ├── download_model.py       fetch the AWQ weights into models/ (git-ignored)
    ├── src/
    │   ├── config.py           connection settings from env
    │   ├── client.py           httpx client: chat() + chat_stream() (no openai SDK)
    │   ├── sentence_pipeline.py tokens → sentence buffer → asyncio.Queue → consumer
    │   ├── cancellation.py      barge-in: stop generation from another task
    │   └── metrics.py           TTFT, inter-token latency, tokens/sec
    ├── examples/               01 one-shot · 02 stream tokens · 03 stream sentences · 04 bench
    ├── web/chat_test.html      minimal browser SSE streaming chat page
    └── docs/                   launch flags · streaming · double-streaming
```

### Not in the repo (git-ignored)

`venv/`, model weights (`models/`, `*.safetensors`), `.env` and any keys.
Only `*.env.example` (placeholders) is committed.

---

## Licence

Own code and docs: [MIT](LICENSE). vLLM (Apache-2.0) and the Qwen models keep
their own upstream licences.
