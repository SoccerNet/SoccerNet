#!/usr/bin/env python3
"""One-off script: stage the single_camera_games_public.zip (raw SoccerNet-Tracking
videos) from EXRCS, ready for upload to SoccerNet/SN-Tracking-Raw.
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from SoccerNet.Downloader import SoccerNetDownloader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", default=os.path.expanduser("~/soccernet_hf_migration/tracking-raw"))
    parser.add_argument("--password", required=True)
    args = parser.parse_args()

    d = SoccerNetDownloader(LocalDirectory=args.staging_root)
    d.password = args.password
    d.downloadRAWVideo(dataset="SoccerNet-Tracking")
    print("DONE:", args.staging_root)


if __name__ == "__main__":
    main()
