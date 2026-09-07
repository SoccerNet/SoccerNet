#!/usr/bin/env python3
"""One-off script: stage ResNET_TF2 (+PCA512) and the two per-frame JSON
features (player bounding boxes, field calibration) from EXRCS for the full
550-game spotting superset, one feature type at a time so each can be
bulk-uploaded to its own branch of SoccerNet/SN-Features.
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from SoccerNet.Downloader import SoccerNetDownloader

FEATURE_TYPES = {
    "resnet-tf2": ["1_ResNET_TF2.npy", "2_ResNET_TF2.npy"],
    "resnet-tf2-pca512": ["1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"],
    "player-boundingbox-maskrcnn": ["1_player_boundingbox_maskrcnn.json", "2_player_boundingbox_maskrcnn.json"],
    "field-calib-ccbv": ["1_field_calib_ccbv.json", "2_field_calib_ccbv.json"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", default=os.path.expanduser("~/soccernet_hf_migration/extra-features"))
    parser.add_argument("--only", choices=list(FEATURE_TYPES), help="stage only this one feature type")
    args = parser.parse_args()

    d = SoccerNetDownloader(LocalDirectory=args.staging_root)
    d.password = "SoccerNet"

    types_to_run = [args.only] if args.only else list(FEATURE_TYPES)

    for branch_name in types_to_run:
        files = FEATURE_TYPES[branch_name]
        branch_dir = os.path.join(args.staging_root, branch_name)
        print(f"=== staging {branch_name}: {files} -> {branch_dir} ===")
        bd = SoccerNetDownloader(LocalDirectory=branch_dir)
        bd.password = "SoccerNet"
        bd.downloadGames(files=files, split=["train", "valid", "test", "challenge"], task="spotting", verbose=True)
        print(f"=== done {branch_name} ===")

    print("ALL STAGING DONE:", args.staging_root)


if __name__ == "__main__":
    main()
