# Double-streaming: tokens → sentences → downstream

The problem this solves, taken from a live voice interview system.

## The naive version blocks

```python
buf = ""
async for token in chat_stream(messages):
    buf += token
    if ends_sentence(buf):
        await speak(buf)      # <-- TTS round trip: network + synth + delivery
        buf = ""
```

`await speak(buf)` blocks the loop for the **entire** downstream round trip before
vLLM is asked for a single token of the next sentence. With real TTS that gap runs
one to several seconds *per sentence* — audible as choppy, stop-start speech.

## The fix: hand sentences to a queue

```python
queue: asyncio.Queue[str | None] = asyncio.Queue()

async def sender():
    while (item := await queue.get()) is not None:
        await speak(item)          # slow work happens HERE, off the token loop

task = asyncio.create_task(sender())

buf = ""
async for token in chat_stream(messages):
    buf += token
    for sentence in flush_complete(buf):
        await queue.put(sentence)  # returns immediately
    buf = trailing_remainder(buf)
await queue.put(None)
await task
```

Now generation of sentence **N+1** runs concurrently with `speak(N)`. The queue is
FIFO, so sentences still reach the consumer in order. This repo's
`src/sentence_pipeline.stream_sentences(token_source, consume, stop=...)` is
exactly this, engine-agnostic.

## Where to cut a sentence

`src/sentence_pipeline.py` flushes on:

1. **sentence-end punctuation** — `.`, `!`, `?`, `。！？`, optionally followed by a
   closing quote/bracket, then whitespace or end-of-buffer.
2. **a run-on safety valve** — if the buffer with no terminal punctuation reaches
   `RUNON_WORDS` (40) words, flush it anyway so audio keeps flowing.

Earlier the safety valve was 15 words — under the *old* serial design that was the
only way to get audio out sooner, but it chopped normal ~16-word sentences at a
prosodically wrong point ("...worked on as a Senior AI" | "Developer?"). Once
generation and the consumer are pipelined, that pressure is gone, so the threshold
is well above any realistic single sentence.

## Barge-in

`src/cancellation.py`:

```python
barge = BargeIn()
tokens = until_cancelled(chat_stream(messages), barge)     # stops yielding on trigger
full = await stream_sentences(tokens, speak, stop=barge.event)
# elsewhere, when the user starts talking:
barge.trigger()
```

- `until_cancelled` breaks the token loop immediately.
- `stream_sentences` sees `stop` set and drains the queue **without** running
  `consume` on anything still pending — no wasted TTS calls for audio that would
  be cut off anyway.
- Whatever was assembled before the trigger is still returned, so the transcript
  matches what the user actually heard.

## Ordering guarantee

The consumer is a single task pulling from one FIFO queue, so sentence *k* is
always fully handled before sentence *k+1* starts — even though sentence *k+1* was
*generated* while *k* was being handled. If your consumer itself streams (e.g. an
mp3 stream), buffer that fully inside `consume` before returning, so downstream
receivers get one clean unit per sentence.
