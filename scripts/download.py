#!/usr/bin/env python3

import argparse
from pathlib import Path
from huggingface_hub import snapshot_download


def download_model(model_id: str, out_dir: str, revision: str | None = None) -> str:
    local_dir = snapshot_download(
        repo_id=model_id,
        local_dir=out_dir,
        revision=revision,
        local_dir_use_symlinks=False,
    )
    return local_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download model snapshot to local cache directory")
    parser.add_argument("--model_id", required=True, help="Hugging Face repo id, e.g. Qwen/Qwen3-4B")
    parser.add_argument("--out", required=True, help="Output root dir, e.g. data/models/")
    parser.add_argument("--revision", default=None, help="Optional HF revision/tag/commit")
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    local_dir = download_model(args.model_id, args.out, args.revision)
    print(local_dir)


if __name__ == "__main__":
    main()
