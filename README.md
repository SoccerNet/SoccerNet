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

# Download SoccerNet labels
mySoccerNetDownloader.downloadGames(files=["Labels.json"], split=["train", "valid", "test"]) # download labels
mySoccerNetDownloader.downloadGames(files=["Labels-v2.json"], split=["train", "valid", "test"]) # download labels SN v2
mySoccerNetDownloader.downloadGames(files=["Labels-cameras.json"], split=["train", "valid", "test"]) # download labels for camera shot

# Download SoccerNet features
mySoccerNetDownloader.downloadGames(files=["1_ResNET_TF2.npy", "2_ResNET_TF2.npy"], split=["train", "valid", "test"]) # download Features
mySoccerNetDownloader.downloadGames(files=["1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"], split=["train", "valid", "test"]) # download Features reduced with PCA
mySoccerNetDownloader.downloadGames(files=["1_player_boundingbox_maskrcnn.json", "2_player_boundingbox_maskrcnn.json"], split=["train", "valid", "test"]) # download Player Bounding Boxes inferred with MaskRCNN
mySoccerNetDownloader.downloadGames(files=["1_field_calib_ccbv.json", "2_field_calib_ccbv.json"], split=["train", "valid", "test"]) # download Field Calibration inferred with CCBV
mySoccerNetDownloader.downloadGames(files=["1_baidu_soccer_embeddings.npy", "2_baidu_soccer_embeddings.npy"], split=["train", "valid", "test"]) # download Frame Embeddings from https://github.com/baidu-research/vidpress-sports

# Download SoccerNet Challenge set (require password from NDA to download videos)
mySoccerNetDownloader.downloadGames(files=["1_ResNET_TF2.npy", "2_ResNET_TF2.npy"], split=["challenge"]) # download ResNET Features
mySoccerNetDownloader.downloadGames(files=["1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"], split=["challenge"]) # download ResNET Features reduced with PCA
mySoccerNetDownloader.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], split=["challenge"]) # download 224p Videos (require password from NDA)
mySoccerNetDownloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv"], split=["challenge"]) # download 720p Videos (require password from NDA)
mySoccerNetDownloader.downloadGames(files=["1_player_boundingbox_maskrcnn.json", "2_player_boundingbox_maskrcnn.json"], split=["challenge"]) # download Player Bounding Boxes inferred with MaskRCNN
mySoccerNetDownloader.downloadGames(files=["1_field_calib_ccbv.json", "2_field_calib_ccbv.json"], split=["challenge"]) # download Field Calibration inferred with CCBV
mySoccerNetDownloader.downloadGames(files=["1_baidu_soccer_embeddings.npy", "2_baidu_soccer_embeddings.npy"], split=["challenge"]) # download Frame Embeddings from https://github.com/baidu-research/vidpress-sports

# Download development kit per task
mySoccerNetDownloader.downloadDataTask(task="calibration-2023", split=["train", "valid", "test", "challenge"])
mySoccerNetDownloader.downloadDataTask(task="caption-2023", split=["train", "valid", "test", "challenge"])
mySoccerNetDownloader.downloadDataTask(task="jersey-2023", split=["train", "test", "challenge"])
mySoccerNetDownloader.downloadDataTask(task="reid-2023", split=["train", "valid", "test", "challenge"])
mySoccerNetDownloader.downloadDataTask(task="spotting-2023", split=["train", "valid", "test", "challenge"])
mySoccerNetDownloader.downloadDataTask(task="spotting-ball-2023", split=["train", "valid", "test", "challenge"], password=<PW_FROM_NDA>)
mySoccerNetDownloader.downloadDataTask(task="tracking-2023", split=["train", "test", "challenge"])
mySoccerNetDownloader.downloadDataTask(task="SpiideoSynLoc", split=["train","valid","test","challenge"]) # 4K Images
mySoccerNetDownloader.downloadDataTask(task="SpiideoSynLoc", split=["train","valid","test","challenge"], version="fullhd") # FullHD Images

# Download SoccerNet videos (require password from NDA to download videos)
mySoccerNetDownloader.password = "Password for videos? (contact the author)"
mySoccerNetDownloader.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], split=["train", "valid", "test"]) # download 224p Videos
mySoccerNetDownloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv"], split=["train", "valid", "test"]) # download 720p Videos
mySoccerNetDownloader.downloadRAWVideo(dataset="SoccerNet") # download 720p Videos
mySoccerNetDownloader.downloadRAWVideo(dataset="SoccerNet-Tracking") # download single camera RAW Videos

# Download SoccerNet in OSL ActionSpotting format
mySoccerNetDownloader.downloadDataTask(task="spotting-OSL", split=["train", "valid", "test", "challenge"], version="ResNET_PCA512")
mySoccerNetDownloader.downloadDataTask(task="spotting-OSL", split=["train", "valid", "test", "challenge"], version="baidu_soccer_embeddings")
mySoccerNetDownloader.downloadDataTask(task="spotting-OSL", split=["train", "valid", "test", "challenge"], version="224p", password=<PW_FROM_NDA>)

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
