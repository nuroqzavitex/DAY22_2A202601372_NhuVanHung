#!/usr/bin/env python3
"""Helper script to upload Lab 22 trained adapters and GGUF to Hugging Face Hub."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    from huggingface_hub import HfApi
except ImportError:
    print("Error: huggingface_hub is required. Install via: pip install huggingface_hub")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Upload Lab 22 artifacts to Hugging Face Hub")
    parser.add_argument("--username", required=True, help="Your Hugging Face username or organization")
    parser.add_argument("--include-gguf", action="store_true", help="Also upload GGUF quant file (optional)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    api = HfApi()

    # 1. Upload SFT adapter
    sft_dir = repo_root / "adapters" / "sft-mini"
    if sft_dir.exists() and any(sft_dir.iterdir()):
        sft_repo = f"{args.username}/lab22-sft-mini-adapter"
        print(f"\n==> Uploading SFT adapter to {sft_repo}...")
        api.create_repo(repo_id=sft_repo, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=str(sft_dir),
            repo_id=sft_repo,
            repo_type="model",
            commit_message="Upload Lab 22 SFT-mini LoRA adapter",
        )
        print(f"✓ SFT adapter uploaded: https://huggingface.co/{sft_repo}")
    else:
        print("⚠ SFT adapter directory not found or empty.")

    # 2. Upload DPO adapter
    dpo_dir = repo_root / "adapters" / "dpo"
    if dpo_dir.exists() and any(dpo_dir.iterdir()):
        dpo_repo = f"{args.username}/lab22-dpo-adapter"
        print(f"\n==> Uploading DPO adapter to {dpo_repo}...")
        api.create_repo(repo_id=dpo_repo, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=str(dpo_dir),
            repo_id=dpo_repo,
            repo_type="model",
            commit_message="Upload Lab 22 DPO LoRA adapter",
        )
        print(f"✓ DPO adapter uploaded: https://huggingface.co/{dpo_repo}")
    else:
        print("⚠ DPO adapter directory not found or empty.")

    # 3. Upload GGUF (optional)
    if args.include_gguf:
        gguf_dir = repo_root / "gguf"
        if gguf_dir.exists() and list(gguf_dir.glob("*.gguf")):
            gguf_repo = f"{args.username}/lab22-dpo-gguf"
            print(f"\n==> Uploading GGUF to {gguf_repo}...")
            api.create_repo(repo_id=gguf_repo, repo_type="model", exist_ok=True)
            api.upload_folder(
                folder_path=str(gguf_dir),
                repo_id=gguf_repo,
                repo_type="model",
                commit_message="Upload Lab 22 DPO GGUF model",
            )
            print(f"✓ GGUF model uploaded: https://huggingface.co/{gguf_repo}")
        else:
            print("⚠ GGUF directory not found or no .gguf files found.")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()