#!/usr/bin/env python3
"""One-off script: stage Labels.json, Labels-cameras.json (generic password)
and Labels-v3.json (reviewers password) for the 500 v1 games, ready for a
bulk upload_folder into SoccerNet/SN-Labels (main branch, flat alongside the
already-migrated Labels-v2.json / Labels-caption.json).
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.utils import getListGames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", default=os.path.expanduser("~/soccernet_hf_migration/remaining-labels"))
    args = parser.parse_args()

    games = getListGames("v1", "spotting")
    print(f"{len(games)} v1 games")

    d = SoccerNetDownloader(LocalDirectory=args.staging_root)

    d.password = "SoccerNet"
    for i, game in enumerate(games):
        d.downloadGame(game=game, files=["Labels.json", "Labels-cameras.json"], spl="train", source="EXRCSDrive")
        if (i + 1) % 50 == 0:
            print(f"  Labels.json/Labels-cameras.json: {i + 1}/{len(games)} done")

    d.password = "SoccerNet_Reviewers_SDATA"
    for i, game in enumerate(games):
        d.downloadGame(game=game, files=["Labels-v3.json"], spl="train", source="EXRCSDrive")
        if (i + 1) % 50 == 0:
            print(f"  Labels-v3.json: {i + 1}/{len(games)} done")

    print("STAGING DONE:", args.staging_root)


if __name__ == "__main__":
    main()
