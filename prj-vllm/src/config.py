"""vLLM connection settings, read from the environment (.env supported)."""
import os

try:  # optional convenience — not required
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_SERVED_MODEL_NAME: str = os.getenv("VLLM_SERVED_MODEL_NAME", "qwen-14b-awq")

# Only set if the server was started with --api-key. Blank -> no auth header.
VLLM_API_KEY: str | None = os.getenv("VLLM_API_KEY") or None

REQUEST_TIMEOUT: float = float(os.getenv("VLLM_TIMEOUT", "60"))
