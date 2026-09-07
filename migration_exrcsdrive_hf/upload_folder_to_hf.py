#!/usr/bin/env python3
"""Bulk-upload a local directory tree to a Hugging Face dataset repo via
upload_folder (not upload_to_hf.py's per-file upload_file loop) - use this
when a task has hundreds/thousands of small files (e.g. per-game features)
rather than a handful of archives.

Usage:
    python upload_folder_to_hf.py --repo-id SoccerNet/SN-Features --visibility public \
        --local-dir ~/soccernet_hf_migration/features-and-labels \
        --allow-patterns "*.npy"

    python upload_folder_to_hf.py --repo-id SoccerNet/SN-Labels --visibility public \
        --local-dir ~/soccernet_hf_migration/features-and-labels \
        --allow-patterns "*/Labels-v2.json" "*/Labels-caption.json"
"""
import argparse

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--visibility", choices=["public", "private", "gated"], required=True)
    parser.add_argument("--branch", default="main")
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--allow-patterns", nargs="+", required=True)
    args = parser.parse_args()

    api = HfApi()
    private = args.visibility == "private"
    gated = args.visibility == "gated"

    api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, private=private, exist_ok=True)
    api.update_repo_settings(repo_id=args.repo_id, repo_type=args.repo_type, gated="manual" if gated else False)

    info = api.repo_info(repo_id=args.repo_id, repo_type=args.repo_type)
    live_gated = bool(info.gated) if info.gated else False
    if info.private != private or live_gated != gated:
        raise SystemExit(
            f"Refusing to upload: live repo private={info.private} gated={info.gated} does not match "
            f"requested visibility={args.visibility}. Check the repo settings on "
            f"huggingface.co/{args.repo_id} before retrying."
        )

    api.upload_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.branch,
        folder_path=args.local_dir,
        allow_patterns=args.allow_patterns,
    )
    print("Done.")


if __name__ == "__main__":
    main()
