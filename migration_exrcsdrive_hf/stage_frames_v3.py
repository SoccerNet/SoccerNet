#!/usr/bin/env python3
"""One-off script: stage + upload Frames-v3.zip for the 500 v1 games (train+
valid+test) to a new "frames-v3" branch of SoccerNet/SoccerNet_raw_HQ.
Frames-v3.zip does not exist for challenge-split games (confirmed via 404).
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.utils import getListGames
from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", default=os.path.expanduser("~/soccernet_hf_migration/frames-v3"))
    args = parser.parse_args()

    games = getListGames("v1", "spotting")
    print(f"{len(games)} v1 games")

    d = SoccerNetDownloader(LocalDirectory=args.staging_root)
    d.password = "SoccerNet_Reviewers_SDATA"
    for i, game in enumerate(games):
        # v1 games use spl="train" (or valid/test) internally only to select the
        # bucket; Frames-v3.zip's token/password is identical across all of
        # train/valid/test/challenge, so any non-"challenge" spl routes correctly.
        d.downloadGame(game=game, files=["Frames-v3.zip"], spl="train", verbose=False)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(games)} games done")

    print("=== uploading frames-v3 to SoccerNet/SoccerNet_raw_HQ ===")
    api = HfApi()
    api.upload_folder(
        repo_id="SoccerNet/SoccerNet_raw_HQ",
        repo_type="dataset",
        revision="frames-v3",
        folder_path=args.staging_root,
        allow_patterns=["*.zip"],
    )
    print("DONE")


if __name__ == "__main__":
    main()
