"""Download the AWQ model into ./models/ (git-ignored).

    pip install huggingface_hub
    python download_model.py

Env:
    MODEL_REPO   Hugging Face repo id  (default: Qwen/Qwen2.5-14B-Instruct-AWQ)
    MODEL_PATH   local destination     (default: ./models/qwen-14b-awq)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("MODEL_REPO", "Qwen/Qwen2.5-14B-Instruct-AWQ")
DEST = os.environ.get("MODEL_PATH", os.path.join(HERE, "models", "qwen-14b-awq"))


def main() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("Install first:  pip install huggingface_hub")

    os.makedirs(DEST, exist_ok=True)
    print(f"downloading {REPO}\n         -> {DEST}")
    snapshot_download(repo_id=REPO, local_dir=DEST)
    print("done. Point start_vllm.sh at this path via MODEL_PATH.")


if __name__ == "__main__":
    main()
