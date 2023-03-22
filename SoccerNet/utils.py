
from pathlib import Path
import json
import os

def getListGames(split="v1", task="spotting", dataset="SoccerNet"):
   

    #  if single split, convert into a list
    if not isinstance(split, list):
        split = [split]

    # if an element is "v1", convert to  train/valid/test
    if "all" in split:
        split = ["train", "valid", "test", "challenge"]
        if task == "frames":
            split = ["train", "valid", "test"]
    if "v1" in split:
        split.pop(split.index("v1"))
        split.append("train")
        split.append("valid")
        split.append("test")

    # if task == "spotting":
    
    listgames = []
    # print(split)
    # loop over splits
    for spl in split: 
        if spl not in ["train", "valid", "test", "challenge"]:
            print(f"[ERROR] unknown split '{spl}'. Please use 'train', 'valid', 'test' or 'challenge' instead.")
            continue
        if dataset == "SoccerNet":
            if task == "spotting":
                if spl == "train":
                    jsonGamesFile = Path(__file__).parent / "data/SoccerNetGamesTrain.json"
                elif spl == "valid":
                    jsonGamesFile = Path(__file__).parent / "data/SoccerNetGamesValid.json"
                elif spl == "test":
                    jsonGamesFile = Path(__file__).parent / "data/SoccerNetGamesTest.json"
                elif spl == "challenge":
                    jsonGamesFile = Path(__file__).parent / "data/SoccerNetGamesChallenge.json"

            elif task == "spotting-ball":
                if spl == "train":
                    jsonGamesFile = Path(__file__).parent / "data/BallTrain.json"
                elif spl == "valid":
                    jsonGamesFile = Path(__file__).parent / "data/BallValid.json"
                elif spl == "test":
                    jsonGamesFile = Path(__file__).parent / "data/BallTest.json"
                elif spl == "challenge":
                    jsonGamesFile = Path(__file__).parent / "data/BallChallenge.json"


            if task == "caption":
                if spl == "train":
                    jsonGamesFile = Path(__file__).parent / "data/SoccerNetCaptionTrain.json"
                elif spl == "valid":
                    jsonGamesFile = Path(__file__).parent / "data/SoccerNetCaptionValid.json"
                elif spl == "test":
                    jsonGamesFile = Path(__file__).parent / "data/SoccerNetCaptionTest.json"
                elif spl == "challenge":
                    jsonGamesFile = Path(__file__).parent / "data/SoccerNetCaptionChallenge.json"

            elif task == "camera-changes":
                if spl == "train":
                    jsonGamesFile = Path(__file__).parent / "data/SoccerNetCameraChangesTrain.json"
                elif spl == "valid":
                    jsonGamesFile = Path(__file__).parent / "data/SoccerNetCameraChangesValid.json"
                elif spl == "test":
                    jsonGamesFile = Path(__file__).parent / "data/SoccerNetCameraChangesTest.json"
                elif spl == "challenge":
                    jsonGamesFile = Path(__file__).parent / "data/SoccerNetCameraChangesChallenge.json"


            elif task == "frames":
                if spl == "train":
                    jsonGamesFile = Path(__file__).parent / "data/SoccerNetFramesTrain.json"
                elif spl == "valid":
                    jsonGamesFile = Path(__file__).parent / "data/SoccerNetFramesValid.json"
                elif spl == "test":
                    jsonGamesFile = Path(__file__).parent / "data/SoccerNetFramesTest.json"
                elif spl == "challenge":
                    jsonGamesFile = Path(__file__).parent / "data/SoccerNetFramesChallenge.json"


            with open(jsonGamesFile, "r") as json_file:
                dictionary = json.load(json_file)

            for championship in dictionary:
                for season in dictionary[championship]:
                    for game in dictionary[championship][season]:

                        listgames.append(os.path.join(championship, season, game))
        
        elif "Headers" in dataset:
            if task == "spotting":
                if spl == "train":
                    jsonGamesFile = Path(__file__).parent / \
                        "data/HeadersTrain.json"
                elif spl == "valid":
                    jsonGamesFile = Path(__file__).parent / \
                        "data/HeadersValid.json"
                elif spl == "test":
                    jsonGamesFile = Path(__file__).parent / \
                        "data/HeadersTest.json"
                elif spl == "challenge":
                    jsonGamesFile = Path(__file__).parent / \
                        "data/HeadersChallenge.json"

            with open(jsonGamesFile, "r") as json_file:
                dictionary = json.load(json_file)

                for game in dictionary:
                    listgames.append(game)

        elif "Ball" in dataset:
            if task == "spotting":
                if spl == "train":
                    jsonGamesFile = Path(__file__).parent / \
                        "data/BallTrain.json"
                elif spl == "valid":
                    jsonGamesFile = Path(__file__).parent / \
                        "data/BallValid.json"
                elif spl == "test":
                    jsonGamesFile = Path(__file__).parent / \
                        "data/BallTest.json"
                elif spl == "challenge":
                    jsonGamesFile = Path(__file__).parent / \
                        "data/BallChallenge.json"

            with open(jsonGamesFile, "r") as json_file:
                dictionary = json.load(json_file)

            for championship in dictionary:
                for season in dictionary[championship]:
                    for game in dictionary[championship][season]:

                        listgames.append(os.path.join(championship, season, game))

    return listgames


if __name__ == "__main__":
    print(len(getListGames(["v1"],task="camera-changes")))
