# Token streaming from vLLM

vLLM's OpenAI server streams with **Server-Sent Events** when the request body has
`"stream": true`.

## Wire format

The response is `Content-Type: text/event-stream`. Each event is a line:

```
data: {"id":"...","choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}
data: {"id":"...","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":null}]}
data: {"id":"...","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
data: [DONE]
```

- Every data line is one JSON object **except** the terminator `data: [DONE]`.
- The token text is `choices[0].delta.content`. The **first** delta usually carries
  `role: "assistant"` and no content; the **last** carries `finish_reason` and no
  content. Skip empty content.
- Blank lines and any `: keep-alive` comment lines can appear — ignore them.

## Client loop (this repo's `src/client.py`)

```python
async with httpx.AsyncClient(timeout=60) as client:
    async with client.stream("POST", url, json=payload, headers=headers) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                token = json.loads(data)["choices"][0]["delta"].get("content", "")
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            if token:
                yield token
```

Points that matter:

- **`client.stream(...)` + `aiter_lines()`**, not `await client.post(...)` — the
  latter buffers the whole response and you lose the point of streaming.
- **`raise_for_status()` inside the `async with`** — a 4xx/5xx still has a body you
  want in the exception.
- **Tolerate parse failures** (`continue`, don't raise) — a half-flushed chunk or a
  keep-alive line should not kill the stream.
- **`timeout`** is the *total* httpx timeout. For long generations set it high, or
  pass `httpx.Timeout(connect=5, read=None, ...)` so only the connect phase is
  bounded.

## Non-streaming

Same endpoint, `"stream": false`, read `choices[0].message.content` from the JSON
body. Use it for scoring / classification / structured extraction where you need
the whole answer before acting and latency-to-first-token doesn't matter.

## Measuring it

`src/metrics.py` wraps a token iterator and records:
- **TTFT** — time from request to the first non-empty token (the number users feel).
- **inter-token gap** — mean ms between tokens after the first.
- **tokens/sec** — total tokens / wall time.

`examples/02_stream_tokens.py` prints these for one call;
`examples/04_bench_ttft.py` sweeps concurrency 1 → 8.
