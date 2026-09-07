<div align="center">
  <img src="https://raw.githubusercontent.com/soccernet/soccernet/main/doc/images/soccernet.png">
</div>

[![Python](https://img.shields.io/pypi/pyversions/SoccerNet)](https://img.shields.io/pypi/pyversions/SoccerNet)
[![Pypi](https://img.shields.io/pypi/v/SoccerNet)](https://pypi.org/project/SoccerNet/)
[![Downloads](https://static.pepy.tech/personalized-badge/SoccerNet?period=month&units=international_system&left_color=grey&right_color=brightgreen&left_text=PyPI%20downloads/month)](https://pepy.tech/project/SoccerNet)
[![Downloads](https://static.pepy.tech/personalized-badge/SoccerNet?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/SoccerNet)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/SoccerNet/SoccerNet/blob/master/LICENSE)
<!-- [![LOC](https://sloc.xyz/github/SoccerNet/SoccerNet/?category=code)](https://github.com/SoccerNet/SoccerNet/) -->
<!-- [![Forks](https://img.shields.io/github/forks/SoccerNet/SoccerNet.svg)](https://github.com/SoccerNet/SoccerNet/network) -->
<!-- [![Issues](https://img.shields.io/github/issues/SoccerNet/SoccerNet.svg)](https://github.com/SoccerNet/SoccerNet/issues) -->
<!-- [![Project Status](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active) -->

# SoccerNet package

```bash
conda create -n SoccerNet python pip
conda activate SoccerNet
pip install SoccerNet
# pip install -e https://github.com/SoccerNet/SoccerNet
# pip install -e .
```

## Data now hosted on Hugging Face

SoccerNet data has moved off the legacy KAUST ownCloud drive (EXRCS) onto the
[`SoccerNet` organization on Hugging Face](https://huggingface.co/SoccerNet).
`downloadDataTask`, `downloadGame`, and `downloadGames` all fetch from Hugging
Face by default (`source="HuggingFace"`); pass `source="EXRCSDrive"` to fall
back to the legacy drive for anything not yet migrated. Call signatures and
local folder layout are unchanged either way, and `password=` is accepted but
ignored (with a warning) wherever access is now controlled by your Hugging
Face account instead — run `huggingface-cli login` once, and request access
on the dataset page for anything gated.

### Datasets

| Dataset | Hugging Face repo | Access |
|---|---|---|
| Camera Calibration 2023 | [`SN-Calibration-2023`](https://huggingface.co/datasets/SoccerNet/SN-Calibration-2023) | public |
| Camera Calibration (legacy, pre-2023) | [`SN-Calibration`](https://huggingface.co/datasets/SoccerNet/SN-Calibration) | public |
| Re-Identification 2023 | [`SN-ReID-2023`](https://huggingface.co/datasets/SoccerNet/SN-ReID-2023) | public |
| Re-Identification (legacy) | same data as `SN-ReID-2023` | public |
| Player Tracking 2023 | [`SN-Tracking-2023`](https://huggingface.co/datasets/SoccerNet/SN-Tracking-2023) | public |
| Player Tracking (legacy) | same data as `SN-Tracking-2023` | public |
| Jersey Number Recognition 2023 | [`SN-Jersey-2023`](https://huggingface.co/datasets/SoccerNet/SN-Jersey-2023) | public |
| Ball Action Spotting 2023 | [`SN-BAS-2023`](https://huggingface.co/datasets/SoccerNet/SN-BAS-2023) | **gated** |
| Ball Action Spotting 2024 | [`SN-BAS-2024`](https://huggingface.co/datasets/SoccerNet/SN-BAS-2024) | public |
| Ball Action Spotting 2025 | [`SN-BAS-2025`](https://huggingface.co/datasets/SoccerNet/SN-BAS-2025) | public |
| Action Spotting (all editions) | per-game features/labels — see below | public |
| Dense Video Captioning 2023 / 2024 | per-game features/labels — see below | public |
| Foul Recognition (MVFouls) 2024 | [`SN-MVFouls-2024`](https://huggingface.co/datasets/SoccerNet/SN-MVFouls-2024) | public |
| Foul Recognition (MVFouls) 2025 | [`SN-MVFouls-2025`](https://huggingface.co/datasets/SoccerNet/SN-MVFouls-2025) | public |
| Game State Reconstruction 2024 | [`SN-GSR-2024`](https://huggingface.co/datasets/SoccerNet/SN-GSR-2024) | public |
| Game State Reconstruction 2025 | [`SN-GSR-2025`](https://huggingface.co/datasets/SoccerNet/SN-GSR-2025) | public |
| Monocular Depth Estimation (football / basketball) | [`SN-Depth`](https://huggingface.co/datasets/SoccerNet/SN-Depth) (branches `football`, `basketball`) | public |
| Monocular Depth Estimation 2025 | [`SN-Depth-2025`](https://huggingface.co/datasets/SoccerNet/SN-Depth-2025) | public |
| Pre-extracted per-game features (all editions) | [`SN-Features`](https://huggingface.co/datasets/SoccerNet/SN-Features) — one branch per feature type: `baidu-soccer-embeddings`, `resnet-tf2`, `resnet-tf2-pca512`, `player-boundingbox-maskrcnn`, `field-calib-ccbv` | public |
| Per-game labels (all editions) | [`SN-Labels`](https://huggingface.co/datasets/SoccerNet/SN-Labels) — `Labels.json`, `Labels-v2.json`, `Labels-v3.json`, `Labels-cameras.json`, `Labels-caption.json` | public |
| Raw broadcast videos, train/valid/test (224p/720p/HQ + frames/clips) | [`SoccerNet_raw_HQ`](https://huggingface.co/datasets/SoccerNet/SoccerNet_raw_HQ) (branches `videos-224p`, `videos-720p`, `videos-HQ`, `frames-720p-2fps`, `clips-720p-10s`, `frames-v3`) | **gated** |
| Raw broadcast videos, challenge split (224p/720p/HQ/LQ) | [`SoccerNet_raw_HQ_Challenge`](https://huggingface.co/datasets/SoccerNet/SoccerNet_raw_HQ_Challenge) (branches `videos-224p`, `videos-720p`, `videos-HQ`, `videos-LQ`) | **gated** |
| Held-out test/challenge ground truth (all tasks above) | [`SN-GroundTruth`](https://huggingface.co/datasets/SoccerNet/SN-GroundTruth), one folder per task | **private** |
| SpiideoSynLoc (4K/FullHD synchronized multi-camera images) | hosted on Spiideo's own S3, not part of this migration | see `downloadDataTask(task="SpiideoSynLoc")` |

For SoccerNet in OSL Action Spotting format, use [`OpenSportsLab/OSL-SoccerNet`](https://huggingface.co/datasets/OpenSportsLab/OSL-SoccerNet) directly (sharded WebDataset format — `datasets.load_dataset` unpacks the `.tar` shards for you):

```python
from datasets import load_dataset
dataset = load_dataset("OpenSportsLab/OSL-SoccerNet", revision="ResNET_PCA512")  # or "224p", "720p"
```

## Structure of the data data for each game

- SoccerNet main folder
  - Leagues (england_epl/europe_uefa-champions-league/france_ligue-1/...)
    - Seasons (2014-2015/2015-2016/2016-2017)
      - Games (format: "{Date} - {Time} - {HomeTeam} {Score} {AwayTeam}")
        - SoccerNet-v2 - Labels / Manual Annotations
          - **video.ini**: information on start/duration for each half of the game in the HQ video, in second
          - **Labels-v2.json**: Labels from SoccerNet-v2 - action spotting
          - **Labels-cameras.json**: Labels from SoccerNet-v1 - camera shot segmentation

        - SoccerNet-v2 - Videos / Automatically Extracted Features
          - **1_224p.mkv**: 224p video 1st half - timmed with start/duration from HQ video - resolution 224*398 - 25 fps
          - **2_224p.mkv**: 224p video 2nd half - timmed with start/duration from HQ video - resolution 224*398 - 25 fps
          - **1_720p.mkv**: 720p video 1st half - timmed with start/duration from HQ video - resolution 720*1280 - 25 fps
          - **2_720p.mkv**: 720p video 2nd half - timmed with start/duration from HQ video - resolution 720*1280 - 25 fps
          - **1_ResNET_TF2.npy**: ResNET features @2fps for 1st half from SoccerNet-v2, [extracted using TF2](https://github.com/SilvioGiancola/SoccerNetv2-DevKit)
          - **2_ResNET_TF2.npy**: ResNET features @2fps for 2nd half from SoccerNet-v2, [extracted using TF2](https://github.com/SilvioGiancola/SoccerNetv2-DevKit)
          - **1_ResNET_TF2_PCA512.npy**: ResNET features @2fps for 1st half from SoccerNet-v2, [extracted using TF2](https://github.com/SilvioGiancola/SoccerNetv2-DevKit), with dimensionality reduced to 512 using PCA
          - **2_ResNET_TF2_PCA512.npy**: ResNET features @2fps for 2nd half from SoccerNet-v2, [extracted using TF2](https://github.com/SilvioGiancola/SoccerNetv2-DevKit), with dimensionality reduced to 512 using PCA
          - **1_ResNET_5fps_TF2.npy**: ResNET features @5fps for 1st half from SoccerNet-v2, [extracted using TF2](https://github.com/SilvioGiancola/SoccerNetv2-DevKit)
          - **2_ResNET_5fps_TF2.npy**: ResNET features @5fps for 2nd half from SoccerNet-v2, [extracted using TF2](https://github.com/SilvioGiancola/SoccerNetv2-DevKit)
          - **1_ResNET_5fps_TF2_PCA512.npy**: ResNET features @5fps for 1st half from SoccerNet-v2, [extracted using TF2](https://github.com/SilvioGiancola/SoccerNetv2-DevKit), with dimensionality reduced to 512 using PCA
          - **2_ResNET_5fps_TF2_PCA512.npy**: ResNET features @5fps for 2nd half from SoccerNet-v2, [extracted using TF2](https://github.com/SilvioGiancola/SoccerNetv2-DevKit), with dimensionality reduced to 512 using PCA
          - **1_ResNET_25fps_TF2.npy**: ResNET features @25fps for 1st half from SoccerNet-v2, [extracted using TF2](https://github.com/SilvioGiancola/SoccerNetv2-DevKit)
          - **2_ResNET_25fps_TF2.npy**: ResNET features @25fps for 2nd half from SoccerNet-v2, [extracted using TF2](https://github.com/SilvioGiancola/SoccerNetv2-DevKit)
          - **1_player_boundingbox_maskrcnn.json**: Player Bounding Boxes @2fps for 1st half, extracted with MaskRCNN
          - **2_player_boundingbox_maskrcnn.json**: Player Bounding Boxes @2fps for 2nd half, extracted with MaskRCNN
          - **1_field_calib_ccbv.json**: Field Camera Calibration @2fps for 1st half, extracted with CCBV
          - **2_field_calib_ccbv.json**: Field Camera Calibration @2fps for 2nd half, extracted with CCBV
          - **1_baidu_soccer_embeddings.npy**: Frame Embeddings for 1st half from [https://github.com/baidu-research/vidpress-sports](https://github.com/baidu-research/vidpress-sports)
          - **2_baidu_soccer_embeddings.npy**: Frame Embeddings for 2nd half from [https://github.com/baidu-research/vidpress-sports](https://github.com/baidu-research/vidpress-sports)

        - Legacy from SoccerNet-v1
          - **Labels.json**: Labels from SoccerNet-v1 - action spotting for goals/cards/subs only
          - **1_C3D.npy**: C3D features @2fps for 1st half from SoccerNet-v1
          - **2_C3D.npy**: C3D features @2fps for 2nd half from SoccerNet-v1
          - **1_C3D_PCA512.npy**: C3D features @2fps for 1st half from SoccerNet-v1, with dimensionality reduced to 512 using PCA
          - **2_C3D_PCA512.npy**: C3D features @2fps for 2nd half from SoccerNet-v1, with dimensionality reduced to 512 using PCA
          - **1_I3D.npy**: I3D features @2fps for 1st half from SoccerNet-v1
          - **2_I3D.npy**: I3D features @2fps for 2nd half from SoccerNet-v1
          - **1_I3D_PCA512.npy**: I3D features @2fps for 1st half from SoccerNet-v1, with dimensionality reduced to 512 using PCA
          - **2_I3D_PCA512.npy**: I3D features @2fps for 2nd half from SoccerNet-v1, with dimensionality reduced to 512 using PCA
          - **1_ResNET.npy**: ResNET features @2fps for 1st half from SoccerNet-v1
          - **2_ResNET.npy**: ResNET features @2fps for 2nd half from SoccerNet-v1
          - **1_ResNET_PCA512.npy**: ResNET features @2fps for 1st half from SoccerNet-v1, with dimensionality reduced to 512 using PCA
          - **2_ResNET_PCA512.npy**: ResNET features @2fps for 2nd half from SoccerNet-v1, with dimensionality reduced to 512 using PCA


## How to Download Games (Python)

```python
from SoccerNet.Downloader import SoccerNetDownloader

mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="path/to/soccernet")

# Download SoccerNet labels (fetched from SoccerNet/SN-Labels by default)
mySoccerNetDownloader.downloadGames(files=["Labels.json"], split=["train", "valid", "test"]) # download labels
mySoccerNetDownloader.downloadGames(files=["Labels-v2.json"], split=["train", "valid", "test"]) # download labels SN v2
mySoccerNetDownloader.downloadGames(files=["Labels-cameras.json"], split=["train", "valid", "test"]) # download labels for camera shot

# Download SoccerNet features (fetched from SoccerNet/SN-Features by default, one branch per feature type)
mySoccerNetDownloader.downloadGames(files=["1_ResNET_TF2.npy", "2_ResNET_TF2.npy"], split=["train", "valid", "test"]) # download Features
mySoccerNetDownloader.downloadGames(files=["1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"], split=["train", "valid", "test"]) # download Features reduced with PCA
mySoccerNetDownloader.downloadGames(files=["1_player_boundingbox_maskrcnn.json", "2_player_boundingbox_maskrcnn.json"], split=["train", "valid", "test"]) # download Player Bounding Boxes inferred with MaskRCNN
mySoccerNetDownloader.downloadGames(files=["1_field_calib_ccbv.json", "2_field_calib_ccbv.json"], split=["train", "valid", "test"]) # download Field Calibration inferred with CCBV
mySoccerNetDownloader.downloadGames(files=["1_baidu_soccer_embeddings.npy", "2_baidu_soccer_embeddings.npy"], split=["train", "valid", "test"]) # download Frame Embeddings from https://github.com/baidu-research/vidpress-sports

# You can also fetch any of the above directly with huggingface_hub, e.g.:
#   from huggingface_hub import snapshot_download
#   snapshot_download(repo_id="SoccerNet/SN-Features", repo_type="dataset", revision="resnet-tf2-pca512", local_dir="path/to/soccernet")

# Download SoccerNet Challenge set features (fetched from Hugging Face, no password needed)
mySoccerNetDownloader.downloadGames(files=["1_ResNET_TF2.npy", "2_ResNET_TF2.npy"], split=["challenge"]) # download ResNET Features
mySoccerNetDownloader.downloadGames(files=["1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"], split=["challenge"]) # download ResNET Features reduced with PCA
mySoccerNetDownloader.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], split=["challenge"]) # download 224p Videos - gated, request access at https://huggingface.co/datasets/SoccerNet/SoccerNet_raw_HQ_Challenge
mySoccerNetDownloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv"], split=["challenge"]) # download 720p Videos - gated, same repo as above
mySoccerNetDownloader.downloadGames(files=["1_player_boundingbox_maskrcnn.json", "2_player_boundingbox_maskrcnn.json"], split=["challenge"]) # download Player Bounding Boxes inferred with MaskRCNN
mySoccerNetDownloader.downloadGames(files=["1_field_calib_ccbv.json", "2_field_calib_ccbv.json"], split=["challenge"]) # download Field Calibration inferred with CCBV
mySoccerNetDownloader.downloadGames(files=["1_baidu_soccer_embeddings.npy", "2_baidu_soccer_embeddings.npy"], split=["challenge"]) # download Frame Embeddings from https://github.com/baidu-research/vidpress-sports

# Download development kit per task
mySoccerNetDownloader.downloadDataTask(task="calibration-2023", split=["train", "valid", "test", "challenge"])
mySoccerNetDownloader.downloadDataTask(task="caption-2023", split=["train", "valid", "test", "challenge"])
mySoccerNetDownloader.downloadDataTask(task="jersey-2023", split=["train", "test", "challenge"])
mySoccerNetDownloader.downloadDataTask(task="reid-2023", split=["train", "valid", "test", "challenge"])
mySoccerNetDownloader.downloadDataTask(task="spotting-2023", split=["train", "valid", "test", "challenge"])
mySoccerNetDownloader.downloadDataTask(task="spotting-ball-2023", split=["train", "valid", "test", "challenge"]) # gated on Hugging Face - request access at https://huggingface.co/datasets/SoccerNet/SN-BAS-2023, then `huggingface-cli login`
mySoccerNetDownloader.downloadDataTask(task="tracking-2023", split=["train", "test", "challenge"])
mySoccerNetDownloader.downloadDataTask(task="SpiideoSynLoc", split=["train","valid","test","challenge"]) # 4K Images
mySoccerNetDownloader.downloadDataTask(task="SpiideoSynLoc", split=["train","valid","test","challenge"], version="fullhd") # FullHD Images

# Download SoccerNet videos - fetched from SoccerNet/SoccerNet_raw_HQ (train/valid/test)
# or SoccerNet/SoccerNet_raw_HQ_Challenge (challenge) by default; both gated,
# request access on the dataset page then `huggingface-cli login` (no password needed)
mySoccerNetDownloader.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], split=["train", "valid", "test"]) # download 224p Videos
mySoccerNetDownloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv"], split=["train", "valid", "test"]) # download 720p Videos
mySoccerNetDownloader.downloadRAWVideo(dataset="SoccerNet") # download 720p Videos
mySoccerNetDownloader.downloadRAWVideo(dataset="SoccerNet-Tracking") # download single camera RAW Videos - still EXRCS-only, not yet migrated
mySoccerNetDownloader.downloadGame(files=["1_720p.mkv", "2_720p.mkv"], game="europe_uefa-champions-league/2016-2017/2017-04-18 - 21-45 Real Madrid 4 - 2 Bayern Munich") # download video for a single game

# Fall back to the legacy EXRCS drive for anything not yet migrated (pass source="EXRCSDrive")
mySoccerNetDownloader.password = "Password for videos? (contact the author)"
mySoccerNetDownloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv"], split=["train", "valid", "test"], source="EXRCSDrive")

```

For SoccerNet in OSL Action Spotting format, use [`OpenSportsLab/OSL-SoccerNet`](https://huggingface.co/datasets/OpenSportsLab/OSL-SoccerNet) directly:

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="OpenSportsLab/OSL-SoccerNet", repo_type="dataset", revision="ResNET_PCA512", local_dir="path/to/soccernet")
```

## How to read the list Games (Python)

```python
from SoccerNet.utils import getListGames
print(getListGames(split="train")) # return list of games recommended for training
print(getListGames(split="valid")) # return list of games recommended for validation
print(getListGames(split="test")) # return list of games recommended for testing
print(getListGames(split="challenge")) # return list of games recommended for challenge
print(getListGames(split=["train", "valid", "test", "challenge"])) # return list of games for training, validation and testing
print(getListGames(split="v1")) # return list of games from SoccerNetv1 (train/valid/test)
```
