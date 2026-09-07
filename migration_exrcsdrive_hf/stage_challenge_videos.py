#!/usr/bin/env python3
"""One-off script: stage + upload the 50 challenge-split games' raw videos to
SoccerNet/SoccerNet_raw_HQ_Challenge, one video quality/branch at a time so
peak local disk usage stays bounded (each quality is uploaded and deleted
before the next one starts) rather than holding all ~345GB at once.
"""
import argparse
import os
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.utils import getListGames
from huggingface_hub import HfApi

QUALITIES = {
    "videos-HQ": ["1_HQ.mkv", "2_HQ.mkv", "video.ini"],
    "videos-720p": ["1_720p.mkv", "2_720p.mkv"],
    "videos-224p": ["1_224p.mkv", "2_224p.mkv"],
    "videos-LQ": ["1.mkv", "2.mkv"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", default=os.path.expanduser("~/soccernet_hf_migration/challenge-videos"))
    parser.add_argument("--password", required=True, help="real NDA password for challenge videos")
    parser.add_argument("--only", choices=list(QUALITIES), help="stage+upload only this one branch")
    args = parser.parse_args()

    games = getListGames("challenge", "spotting")
    print(f"{len(games)} challenge games")

    api = HfApi()
    branches = [args.only] if args.only else list(QUALITIES)

    for branch in branches:
        files = QUALITIES[branch]
        branch_dir = os.path.join(args.staging_root, branch)
        print(f"=== downloading {branch}: {files} for {len(games)} games -> {branch_dir} ===")
        d = SoccerNetDownloader(LocalDirectory=branch_dir)
        d.password = args.password
        for i, game in enumerate(games):
            # downloadGame defaults to source="HuggingFace" now - force EXRCSDrive
            # here since this script's whole job is staging FROM EXRCS.
            d.downloadGame(game=game, files=files, spl="challenge", verbose=False, source="EXRCSDrive")
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(games)} games done")

        print(f"=== uploading {branch} to SoccerNet/SoccerNet_raw_HQ_Challenge ===")
        api.upload_folder(
            repo_id="SoccerNet/SoccerNet_raw_HQ_Challenge",
            repo_type="dataset",
            revision=branch,
            folder_path=branch_dir,
            allow_patterns=["*.mkv", "*.ini"],
        )
        print(f"=== done {branch}, freeing local disk ===")
        shutil.rmtree(branch_dir, ignore_errors=True)

    print("ALL CHALLENGE VIDEO BRANCHES DONE")


if __name__ == "__main__":
    main()
