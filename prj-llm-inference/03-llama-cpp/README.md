# 03-llama-cpp

Run a quantized **GGUF** model with **llama.cpp** through `llama-cpp-python`.
Works on CPU; offload layers to the GPU with `--gpu-layers -1`.

```bash
python -m venv venv && venv\Scripts\pip install -r requirements.txt
# put a .gguf file in models/ (git-ignored), e.g. a Q4_K_M quant of Qwen2.5-0.5B-Instruct
python chat.py --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf "Hi"
python chat.py --model models/<file>.gguf --stream "Write a haiku"
```

- Pros: tiny memory footprint, runs on CPU/Mac, one file per model.
- Cons: a CUDA build needs extra steps; lower throughput than vLLM.

## Use cases
Laptops, CPU-only machines, Macs and edge devices: an offline assistant, a small
local **agent** that routes tasks, or a private data-extraction tool where RAM is
tight. Quantized GGUF files (Q4_K_M etc.) trade a little quality for a lot of
memory. See the [parent README](../README.md).
