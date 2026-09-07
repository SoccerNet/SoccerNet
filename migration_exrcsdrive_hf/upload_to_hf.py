#!/usr/bin/env python3
"""Upload staged local files to a Hugging Face dataset repo.

Creates the repo if it doesn't exist yet, at the requested visibility, then
refuses to upload if the repo's LIVE visibility on huggingface.co doesn't
match what was requested (catches drift / an already-existing repo with the
wrong setting instead of silently uploading into it).

Usage:
    python upload_to_hf.py --repo-id SoccerNet/SN-Calibration --visibility public \
        --local-dir ~/soccernet_hf_migration/calibration-2023 \
        --files train.zip valid.zip test.zip challenge.zip \
        --path-prefix 2023

    python upload_to_hf.py --repo-id SoccerNet/SN-GroundTruth --visibility private \
        --local-dir ~/soccernet_hf_migration/calibration-2023 \
        --files test_secret.zip challenge_secret.zip \
        --path-prefix calibration-2023
"""
import argparse
import os

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True, help="e.g. SoccerNet/SN-Calibration")
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--visibility", choices=["public", "private", "gated"], required=True)
    parser.add_argument("--branch", default="main")
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--files", nargs="+", required=True, help="filenames inside --local-dir to upload")
    parser.add_argument("--path-prefix", default="", help="folder inside the repo, e.g. 2023")
    args = parser.parse_args()

    api = HfApi()
    # "gated" is a public repo with an access-request gate, not a private repo -
    # only "private" actually hides the repo from non-members.
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

    for fname in args.files:
        local_path = os.path.join(args.local_dir, fname)
        path_in_repo = os.path.join(args.path_prefix, fname) if args.path_prefix else fname
        print(f"uploading {local_path} -> {args.repo_id}:{path_in_repo}")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.branch,
        )

    print("Done.")


if __name__ == "__main__":
    main()
