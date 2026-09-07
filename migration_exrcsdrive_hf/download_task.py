#!/usr/bin/env python3
"""Stage a SoccerNet task from EXRCS using the existing SoccerNetDownloader.

Deliberately reuses SoccerNet.Downloader.SoccerNetDownloader instead of
hand-rolled HTTP calls, so the staged local layout is always identical to
what the shipped library already produces.

Usage:
    python download_task.py --task calibration-2023 --split train valid test challenge \
        --password SoccerNet
    python download_task.py --task calibration-2023 --split test_labels challenge_labels \
        --password-env SOCCERNET_NDA_PASSWORD
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from SoccerNet.Downloader import SoccerNetDownloader


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, help="task name as understood by downloadDataTask, e.g. calibration-2023")
    parser.add_argument("--split", nargs="+", required=True, help="one or more of train/valid/test/challenge/test_labels/challenge_labels")
    parser.add_argument("--version", default=None, help="task version, e.g. 224p / 720p, if the task has one")
    parser.add_argument("--staging-root", default=os.path.expanduser("~/soccernet_hf_migration"))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--password", help="literal password (fine for the public generic password only)")
    group.add_argument("--password-env", help="name of an env var holding the real NDA/reviewers password")
    args = parser.parse_args()

    password = args.password or os.environ[args.password_env]

    downloader = SoccerNetDownloader(LocalDirectory=args.staging_root)
    downloader.downloadDataTask(
        task=args.task,
        split=args.split,
        password=password,
        version=args.version,
        source="EXRCSDrive",  # this tool stages FROM EXRCS; never rely on the HuggingFace default
    )

    print(f"Staged {args.task} {args.split} under {os.path.join(args.staging_root, args.task)}")


if __name__ == "__main__":
    main()
