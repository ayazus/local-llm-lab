# vLLM launch flags

The command in `start_vllm.sh`, flag by flag.

```bash
python -m vllm.entrypoints.openai.api_server \
    --model ./models/qwen-14b-awq \
    --served-model-name qwen-14b-awq \
    --quantization awq_marlin \
    --dtype float16 \
    --max-model-len 24000 \
    --gpu-memory-utilization 0.85 \
    --host 0.0.0.0 --port 8000
```

| Flag | Purpose | Trade-off |
|---|---|---|
| `--model` | local path or HF repo id of the weights | a local path avoids a re-download on every start |
| `--served-model-name` | the `model` string clients send in the request body | keep it short and stable; **must match** `VLLM_SERVED_MODEL_NAME` |
| `--quantization awq_marlin` | use the Marlin kernels for AWQ 4-bit weights | ~4x smaller than fp16, fast on Ampere+; needs an AWQ checkpoint |
| `--dtype float16` | compute dtype for activations / KV cache | `float16` is safe on consumer GPUs; `bfloat16` on newer cards |
| `--max-model-len` | max prompt + output tokens per request | **the KV cache is sized from this** — lower it to fit a bigger batch or a smaller GPU; raise it only if you truly need long context |
| `--gpu-memory-utilization` | fraction of VRAM vLLM may claim | 0.85 leaves room for other processes; push to 0.90–0.95 if vLLM is alone on the card |
| `--host 0.0.0.0` | listen on all interfaces | use `127.0.0.1` if only local clients connect |
| `--port` | HTTP port | client uses `http://host:PORT/v1` |

## Common extras

| Flag | When |
|---|---|
| `--api-key <value>` | require `Authorization: Bearer <value>` on every request (set the same value in `.env` as `VLLM_API_KEY`) |
| `--tensor-parallel-size N` | split the model across N GPUs |
| `--enforce-eager` | disable CUDA graphs — slower, but lower memory and easier to debug |
| `--max-num-seqs N` | cap concurrent sequences (throughput vs latency) |
| `--enable-prefix-caching` | reuse KV for shared prompt prefixes (big win when every request shares a long system prompt) |

## Sizing intuition

VRAM ≈ quantized weights + KV cache + activations.
- Weights (14B AWQ) ≈ ~9 GB.
- KV cache grows with `max_num_seqs × max_model_len`. Halving `--max-model-len`
  roughly doubles how many requests fit at once.
- If you OOM on startup: lower `--gpu-memory-utilization`, then `--max-model-len`.
- If you OOM mid-run under load: lower `--max-num-seqs`.
