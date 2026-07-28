"""
Utility untuk mengelola versioning adapter checkpoint hasil fine-tuning,
termasuk opsi push ke Hugging Face Hub sebagai private repo.

Contoh pemakaian:
  python checkpoint_manager.py --list
  python checkpoint_manager.py --push adapter_20260728_101500
"""

import os
import argparse
import logging
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("checkpoint_manager")

ADAPTER_DIR = Path(os.getenv("ADAPTER_DIR", "checkpoints"))
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID")  # contoh: "username/llama-3.2-3b-streaming-lora"


def list_checkpoints():
    checkpoints = sorted(ADAPTER_DIR.glob("adapter_*"))
    if not checkpoints:
        logger.info("Belum ada checkpoint tersimpan.")
        return []

    for i, ckpt in enumerate(checkpoints, 1):
        logger.info(f"{i}. {ckpt.name}")
    return checkpoints


def push_checkpoint(checkpoint_path: Path, repo_id: str):
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN belum diset di .env")

    api = HfApi(token=HF_TOKEN)
    create_repo(repo_id, token=HF_TOKEN, private=True, exist_ok=True)

    api.upload_folder(
        folder_path=str(checkpoint_path),
        repo_id=repo_id,
        commit_message=f"Upload checkpoint: {checkpoint_path.name}",
    )
    logger.info(f"Checkpoint '{checkpoint_path.name}' berhasil di-push ke {repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="Tampilkan semua checkpoint tersedia")
    parser.add_argument("--push", type=str, help="Nama folder checkpoint yang akan di-push ke HF Hub")
    args = parser.parse_args()

    if args.list:
        list_checkpoints()
    elif args.push:
        checkpoint_path = ADAPTER_DIR / args.push
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint '{args.push}' tidak ditemukan di {ADAPTER_DIR}")
        if not HF_REPO_ID:
            raise ValueError("HF_REPO_ID belum diset di .env")
        push_checkpoint(checkpoint_path, HF_REPO_ID)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
