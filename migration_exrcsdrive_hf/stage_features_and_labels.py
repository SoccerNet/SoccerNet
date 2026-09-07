#!/usr/bin/env python3
"""One-off script: stage per-game baidu embeddings + v2/caption labels from
EXRCS into a clean <game>/<file> layout, ready for a bulk upload_folder to
SoccerNet/SN-Features and SoccerNet/SN-Labels.

Not a generic reusable tool like the other migration_exrcsdrive_hf scripts -
this is specific to the spotting-2023/caption-2023 per-game migration, which
is structurally different (thousands of small per-game files instead of a
handful of archives).
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from SoccerNet.Downloader import SoccerNetDownloader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", default=os.path.expanduser("~/soccernet_hf_migration/features-and-labels"))
    args = parser.parse_args()

    d = SoccerNetDownloader(LocalDirectory=args.staging_root)
    d.password = "SoccerNet"

    print("Downloading embeddings + Labels-v2.json for the full spotting game list (550 games)...")
    d.downloadGames(
        files=["1_baidu_soccer_embeddings.npy", "2_baidu_soccer_embeddings.npy", "Labels-v2.json"],
        split=["train", "valid", "test", "challenge"],
        task="spotting",
        verbose=True,
    )

    print("Downloading Labels-caption.json for the caption game list (513 games, subset of spotting)...")
    d.downloadGames(
        files=["Labels-caption.json"],
        split=["train", "valid", "test", "challenge"],
        task="caption",
        verbose=True,
    )

    print("STAGING DONE:", args.staging_root)


if __name__ == "__main__":
    main()
