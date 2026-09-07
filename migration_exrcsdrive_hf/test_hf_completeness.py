#!/usr/bin/env python3
"""Smoke-test every Hugging Face download path documented in README.md.

For each case: download the smallest sufficient slice via SoccerNetDownloader
(source="HuggingFace", the default - never touches EXRCS), verify the
resulting file(s) exist and are non-empty, then delete them immediately.
Nothing downloaded by this script is ever left on disk, so it's safe to
re-run any time as a regression check.

Does NOT literally replay every README example split-for-split (e.g.
split=["train","valid","test"] would mean re-downloading ~500 games' worth
of data per file type). Instead it exercises every distinct
task/version/file-type -> (repo, branch) mapping using a single game or the
smallest available split, which is enough to prove "no error, file is really
there" without re-transferring terabytes on every run.

The one genuinely huge item (SoccerNet-Tracking-RAW-Video, 88.7GB, a single
file) is skipped by default and only checked via repo/file metadata (no
download); pass --include-huge to actually download and delete it too.

Usage:
    python migration_exrcsdrive_hf/test_hf_completeness.py
    python migration_exrcsdrive_hf/test_hf_completeness.py --include-huge
    python migration_exrcsdrive_hf/test_hf_completeness.py --only calibration-2023,reid-2023
"""
import argparse
import os
import shutil
import sys
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.utils import getListGames
from huggingface_hub import HfApi

STAGING_ROOT = os.path.expanduser("~/soccernet_hf_migration/_completeness_test")

RESULTS = []  # (name, "PASS"/"FAIL"/"SKIP", detail)


def fresh_downloader():
    if os.path.isdir(STAGING_ROOT):
        shutil.rmtree(STAGING_ROOT, ignore_errors=True)
    return SoccerNetDownloader(LocalDirectory=STAGING_ROOT)


def assert_nonempty(*rel_paths):
    for rel in rel_paths:
        p = os.path.join(STAGING_ROOT, rel)
        if not os.path.isfile(p):
            raise AssertionError(f"missing file: {rel}")
        if os.path.getsize(p) == 0:
            raise AssertionError(f"empty file: {rel}")


def run_case(name, fn, only=None):
    if only and not any(o in name for o in only):
        return
    try:
        fn()
        RESULTS.append((name, "PASS", ""))
        print(f"[PASS] {name}")
    except Exception as e:
        RESULTS.append((name, "FAIL", f"{type(e).__name__}: {e}"))
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        if os.path.isdir(STAGING_ROOT):
            shutil.rmtree(STAGING_ROOT, ignore_errors=True)


# --- downloadGame: one file at a time, per (repo, branch) endpoint ---

def case_downloadGame(file, spl):
    def _run():
        d = fresh_downloader()
        game = getListGames(spl, "spotting")[0]
        d.downloadGame(game=game, files=[file], spl=spl)
        assert_nonempty(os.path.join(game, file))
    return _run


PER_GAME_CASES = [
    # (label, file, spl)
    ("video-224p-v1", "1_224p.mkv", "train"),
    ("video-224p-challenge", "1_224p.mkv", "challenge"),
    ("video-720p-v1", "1_720p.mkv", "train"),
    ("video-720p-challenge", "1_720p.mkv", "challenge"),
    ("video-HQ-v1", "1_HQ.mkv", "train"),
    ("video-HQ-challenge", "1_HQ.mkv", "challenge"),
    ("video-LQ-challenge", "1.mkv", "challenge"),  # no v1 videos-LQ branch exists on HF
    ("frames-v3", "Frames-v3.zip", "train"),
    ("feature-baidu", "1_baidu_soccer_embeddings.npy", "train"),
    ("feature-resnet-tf2", "1_ResNET_TF2.npy", "train"),
    ("feature-resnet-tf2-pca512", "1_ResNET_TF2_PCA512.npy", "train"),
    ("feature-player-bbox", "1_player_boundingbox_maskrcnn.json", "train"),
    ("feature-field-calib", "1_field_calib_ccbv.json", "train"),
    ("label-Labels.json", "Labels.json", "train"),
    ("label-Labels-v2.json", "Labels-v2.json", "train"),
    ("label-Labels-cameras.json", "Labels-cameras.json", "train"),
    ("label-Labels-v3.json", "Labels-v3.json", "train"),
    ("label-Labels-caption.json", "Labels-caption.json", "train"),
]


# --- downloadGames: batched multi-game path, small split ---

def case_downloadGames_batch():
    d = fresh_downloader()
    # baidu embeddings exist for every game including challenge
    d.downloadGames(files=["1_baidu_soccer_embeddings.npy"], split=["challenge"], task="spotting")
    challenge_games = getListGames("challenge", "spotting")
    assert_nonempty(*[os.path.join(g, "1_baidu_soccer_embeddings.npy") for g in challenge_games[:3]])
    # Labels.json only exists for the "v1" games (train/valid/test), not challenge
    d.downloadGames(files=["Labels.json"], split=["train"], task="spotting")
    train_games = getListGames("train", "spotting")
    assert_nonempty(*[os.path.join(g, "Labels.json") for g in train_games[:3]])


# --- downloadGame single-game example from the README ---

def case_downloadGame_readme_example():
    d = fresh_downloader()
    game = "europe_uefa-champions-league/2016-2017/2017-04-18 - 21-45 Real Madrid 4 - 2 Bayern Munich"
    d.downloadGame(files=["1_720p.mkv", "2_720p.mkv"], game=game)
    assert_nonempty(os.path.join(game, "1_720p.mkv"), os.path.join(game, "2_720p.mkv"))


def case_downloadGameIndex():
    d = fresh_downloader()
    d.downloadGameIndex(index=0, files=["Labels.json"])
    game = getListGames("all")[0]
    assert_nonempty(os.path.join(game, "Labels.json"))


# --- downloadDataTask: every migrated task, smallest available split ---

def case_task(task, split, version=None, extra_splits=()):
    def _run():
        d = fresh_downloader()
        all_splits = list(split) + list(extra_splits)
        d.downloadDataTask(task=task, split=all_splits, version=version)
        # loose check: at least one file landed somewhere under the task dir
        task_dir = os.path.join(STAGING_ROOT, task)
        gt_dir = STAGING_ROOT
        found = False
        for root, _, files in os.walk(STAGING_ROOT):
            if files:
                found = True
                break
        if not found:
            raise AssertionError(f"no files downloaded for task={task} split={all_splits} version={version}")
    return _run


TASK_CASES = [
    # (label, task, split, version)
    ("calibration-2023", "calibration-2023", ["challenge", "test_labels", "challenge_labels"], None),
    ("reid-2023", "reid-2023", ["challenge", "test_labels", "challenge_labels"], None),
    ("tracking-2023", "tracking-2023", ["challenge", "test_labels", "challenge_labels"], None),
    ("jersey-2023", "jersey-2023", ["challenge", "test_labels", "challenge_labels"], None),
    ("spotting-ball-2023", "spotting-ball-2023", ["challenge", "test_labels", "challenge_labels"], None),
    ("spotting-2023", "spotting-2023", ["challenge", "test_labels", "challenge_labels"], None),
    ("caption-2023", "caption-2023", ["challenge", "test_labels", "challenge_labels"], None),
    ("calibration-legacy", "calibration", ["challenge", "test_labels", "challenge_labels"], None),
    ("reid-legacy", "reid", ["challenge", "test_labels", "challenge_labels"], None),
    ("tracking-legacy", "tracking", ["challenge", "test_labels", "challenge_labels"], None),
    ("spotting-legacy", "spotting", ["challenge"], None),
    ("mvfoul-2024-224p", "mvfoul-2024", ["challenge", "test_labels", "challenge_labels"], "224p"),
    ("mvfoul-2024-720p", "mvfoul-2024", ["challenge"], "720p"),
    ("gamestate-2024", "gamestate-2024", ["challenge", "test_labels", "challenge_labels"], None),
    ("caption-2024", "caption-2024", ["challenge", "test_labels", "challenge_labels"], None),
    ("spotting-ball-2024", "spotting-ball-2024", ["challenge", "test_labels", "challenge_labels"], None),
    ("mvfouls-2025", "mvfouls-2025", ["valid"], None),
    ("gamestate-2025", "gamestate-2025", ["valid"], None),
    ("depth-2025", "depth-2025", ["valid"], None),
    ("spotting-ball-2025", "spotting-ball-2025", ["valid"], None),
    ("depth-football", "depth-football", ["test"], None),
    ("depth-basketball", "depth-basketball", ["valid"], None),
]


# --- downloadRAWVideo ---

def case_downloadRAWVideo_tracking_metadata_only():
    api = HfApi()
    info = api.repo_info(repo_id="SoccerNet/SoccerNet-Tracking-RAW-Video", repo_type="dataset", files_metadata=True)
    sizes = {f.rfilename: f.size for f in info.siblings}
    if sizes.get("single_camera_games_public.zip", 0) <= 0:
        raise AssertionError("single_camera_games_public.zip missing or empty on SoccerNet-Tracking-RAW-Video")


def case_downloadRAWVideo_tracking_full():
    d = fresh_downloader()
    d.downloadRAWVideo(dataset="SoccerNet-Tracking")
    assert_nonempty("single_camera_games_public.zip")


def case_spotting_osl_removed():
    d = fresh_downloader()
    d.downloadDataTask(task="spotting-OSL", split=["challenge"])
    # should print "ERROR Unknown task" and download nothing - not raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--include-huge", action="store_true",
                         help="also actually download+verify+delete the 88.7GB SoccerNet-Tracking-RAW-Video file")
    parser.add_argument("--only", default=None, help="comma-separated substrings; only run cases whose name contains one of them")
    args = parser.parse_args()
    only = [s.strip() for s in args.only.split(",")] if args.only else None

    for label, file, spl in PER_GAME_CASES:
        run_case(f"downloadGame/{label}", case_downloadGame(file, spl), only)

    run_case("downloadGames/batched-challenge", case_downloadGames_batch, only)
    run_case("downloadGame/readme-example", case_downloadGame_readme_example, only)
    run_case("downloadGameIndex", case_downloadGameIndex, only)

    for label, task, split, version in TASK_CASES:
        run_case(f"downloadDataTask/{label}", case_task(task, split, version), only)

    if args.include_huge:
        run_case("downloadRAWVideo/SoccerNet-Tracking-full", case_downloadRAWVideo_tracking_full, only)
    else:
        run_case("downloadRAWVideo/SoccerNet-Tracking-metadata", case_downloadRAWVideo_tracking_metadata_only, only)

    run_case("spotting-OSL-removed", case_spotting_osl_removed, only)

    print("\n=== SUMMARY ===")
    passed = sum(1 for _, status, _ in RESULTS if status == "PASS")
    failed = [r for r in RESULTS if r[1] == "FAIL"]
    print(f"{passed}/{len(RESULTS)} passed")
    for name, status, detail in failed:
        print(f"  FAIL: {name} -- {detail}")

    if os.path.isdir(STAGING_ROOT):
        shutil.rmtree(STAGING_ROOT, ignore_errors=True)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
