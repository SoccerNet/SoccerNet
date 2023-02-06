
import urllib.request
import os
from tqdm import tqdm
import json
import random
from SoccerNet.utils import getListGames

class MyProgressBar():
    def __init__(self, filename):
        self.pbar = None
        self.filename = filename

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = tqdm(total=total_size, unit='iB', unit_scale=True)
            self.pbar.set_description(f"Downloading {self.filename}...")
            self.pbar.refresh()  # to show immediately the update

        self.pbar.update(block_size)



import uuid
from google_measurement_protocol import event, report

class OwnCloudDownloader():
    def __init__(self, LocalDirectory, OwnCloudServer):
        self.LocalDirectory = LocalDirectory
        self.OwnCloudServer = OwnCloudServer

        self.client_id = uuid.uuid4()

    def downloadFile(self, path_local, path_owncloud, user=None, password=None, verbose=True):
        # return 0: successfully downloaded
        # return 1: HTTPError
        # return 2: unsupported error
        # return 3: file already exist locally
        # return 4: password is None
        # return 5: user is None

        if password is None:
            print(f"password required for {path_local}")
            return 4
        if user is None:
            return 5

        if user is not None or password is not None:  
            # update Password
             
            password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
            password_mgr.add_password(
                None, self.OwnCloudServer, user, password)
            handler = urllib.request.HTTPBasicAuthHandler(
                password_mgr)
            opener = urllib.request.build_opener(handler)
            urllib.request.install_opener(opener)

        if os.path.exists(path_local): # check existence
            if verbose:
                print(f"{path_local} already exists")
            return 2

        try:
            try:
                os.makedirs(os.path.dirname(path_local), exist_ok=True)
                urllib.request.urlretrieve(
                    path_owncloud, path_local, MyProgressBar(path_local))

            except urllib.error.HTTPError as identifier:
                print(identifier)
                return 1
        except:
            os.remove(path_local)
            raise
            return 2

        # record googleanalytics event
        data = event('download', os.path.basename(path_owncloud))
        report('UA-99166333-3', self.client_id, data)

        return 0


class SoccerNetDownloader(OwnCloudDownloader):
    def __init__(self, LocalDirectory,
                 OwnCloudServer="https://exrcsdrive.kaust.edu.sa/exrcsdrive/public.php/webdav/"):
        super(SoccerNetDownloader, self).__init__(
            LocalDirectory, OwnCloudServer)
        self.password = None

    def downloadDataTask(self, task, split=["train","valid","test","challenge"], verbose=True, password="SoccerNet"): # Generic password for public data
        # 2023
        if task == "calibration-2023":
            if "train" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "train.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "train.zip").replace(
                                            ' ', '%20').replace('\\', '/'),
                                        # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/UxCNUvxClQ8Hw2R # user for calibration splits
                                        user="DxdZ39FT9GqCkEe",
                                        password=password,
                                        verbose=verbose)
            if "valid" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "valid.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "valid.zip").replace(
                                            ' ', '%20').replace('\\', '/'),
                                        # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/UxCNUvxClQ8Hw2R # user for calibration splits
                                        user="DxdZ39FT9GqCkEe",
                                        password=password,
                                        verbose=verbose)
            if "test" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "test.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "test.zip").replace(
                                            ' ', '%20').replace('\\', '/'),
                                        # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/UxCNUvxClQ8Hw2R # user for calibration splits
                                        user="DxdZ39FT9GqCkEe",
                                        password=password,
                                        verbose=verbose)
            if "challenge" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "challenge.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "challenge.zip").replace(
                                            ' ', '%20').replace('\\', '/'),
                                        # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/DxdZ39FT9GqCkEe # user for calibration splits
                                        user="DxdZ39FT9GqCkEe",
                                        password=password,
                                        verbose=verbose)
            if "test_labels" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "test_secret.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "test_secret.zip").replace(
                                            ' ', '%20').replace('\\', '/'),
                                        # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/D4xoTmkC7SEJGZp # user for calibration splits GT
                                        user="D4xoTmkC7SEJGZp",
                                        password=password,
                                        verbose=verbose)
            if "challenge_labels" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "challenge_secret.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "challenge_secret.zip").replace(
                                            ' ', '%20').replace('\\', '/'),
                                        # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/D4xoTmkC7SEJGZp # user for calibration splits GT
                                        user="D4xoTmkC7SEJGZp",
                                        password=password,
                                        verbose=verbose)
        elif task == "reid-2023":
            if "train" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "train.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "train.zip").replace(
                                            ' ', '%20').replace('\\', '/'),
                                        # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/UNYXsOv9RJFU3ax # user for reid splits
                                        user="UNYXsOv9RJFU3ax",
                                        password=password,
                                        verbose=verbose)
            if "valid" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "valid.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "valid.zip").replace(
                                            ' ', '%20').replace('\\', '/'),
                                        # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/UNYXsOv9RJFU3ax # user for reid splits
                                        user="UNYXsOv9RJFU3ax",
                                        password=password,
                                        verbose=verbose)
            if "test" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "test.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "test.zip").replace(
                                            ' ', '%20').replace('\\', '/'),
                                        # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/UNYXsOv9RJFU3ax # user for reid splits
                                        user="UNYXsOv9RJFU3ax",
                                        password=password,
                                        verbose=verbose)
            if "challenge" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "challenge.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "challenge.zip").replace(
                                            ' ', '%20').replace('\\', '/'),
                                        # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/UNYXsOv9RJFU3ax # user for reid splits
                                        user="UNYXsOv9RJFU3ax",
                                        password=password,
                                        verbose=verbose)
            if "test_labels" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "test_bbox_info.json"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "test_bbox_info.json").replace(
                                            ' ', '%20').replace('\\', '/'),
                                        # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/GXmA6cNHEJS0MAj # user for reid splits GT
                                        user="GXmA6cNHEJS0MAj",
                                        password=password,
                                        verbose=verbose)
                if res == 1:  # HTTPError
                    print("This function should download test_labels.zip for reid - but test_labels.zip was not uploaded on the server yet! - or check the password")

            if "challenge_labels" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "challenge_bbox_info.json"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "challenge_bbox_info.json").replace(
                                            ' ', '%20').replace('\\', '/'),
                                        # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/GXmA6cNHEJS0MAj # user for reid splits GT
                                        user="GXmA6cNHEJS0MAj",
                                        password=password,
                                        verbose=verbose)
        elif task == "spotting-ball-2023":
            if "train" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "train.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "train.zip").replace(
                                            ' ', '%20').replace('\\', '/'),
                                        # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/A1ncJfjV31lPSTa # user for fine grained spotting splits
                                        user="A1ncJfjV31lPSTa",
                                        password=password,
                                        verbose=verbose)
            if "valid" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "valid.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "valid.zip").replace(
                                            ' ', '%20').replace('\\', '/'),
                                        # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/A1ncJfjV31lPSTa # user for fine grained spotting splits
                                        user="A1ncJfjV31lPSTa",
                                        password=password,
                                        verbose=verbose)
            if "test" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "test.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "test.zip").replace(
                                            ' ', '%20').replace('\\', '/'),
                                        # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/A1ncJfjV31lPSTa # user for fine grained spotting splits
                                        user="A1ncJfjV31lPSTa",
                                        password=password,
                                        verbose=verbose)
            if "challenge" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "challenge.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "challenge.zip").replace(
                                            ' ', '%20').replace('\\', '/'),
                                        # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/A1ncJfjV31lPSTa # user for fine grained spotting splits
                                        user="A1ncJfjV31lPSTa",
                                        password=password,
                                        verbose=verbose)
            if "test_labels" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "test-private.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "test-private.zip").replace(
                                            ' ', '%20').replace('\\', '/'),
                                        # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/s3Rub1T7C81b90o # user for fine grained spotting splits GT
                                        user="s3Rub1T7C81b90o",
                                        password=password,
                                        verbose=verbose)
                if res == 1:  # HTTPError
                    print("This function should download test_labels.zip for fine grained spotting - but test_labels.zip was not uploaded on the server yet! - or check the password")

            if "challenge_labels" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "challenge-private.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "challenge-private.zip").replace(
                                            ' ', '%20').replace('\\', '/'),
                                        # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/s3Rub1T7C81b90o # user for fine grained spotting splits GT
                                        user="s3Rub1T7C81b90o",
                                        password=password,
                                        verbose=verbose)
        elif task == "tracking-2023":
            if "train" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "train.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "train.zip").replace(
                                            ' ', '%20').replace('\\', '/'),
                                        # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/OP7fl7h25NqGfcN # user for tracking splits
                                        user="OP7fl7h25NqGfcN",
                                        password=password,
                                        verbose=verbose)
                if res == 1:  # HTTPError
                    print(
                        "This function should download train.zip for tracking - but train.zip was not uploaded on the server yet!")

            if "test" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "test.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "test.zip").replace(
                                            ' ', '%20').replace('\\', '/'),
                                        # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/OP7fl7h25NqGfcN # user for tracking splits
                                        user="OP7fl7h25NqGfcN",
                                        password=password,
                                        verbose=verbose)
                if res == 1:  # HTTPError
                    print(
                        "This function should download test.zip for tracking - but test.zip was not uploaded on the server yet!")

            if "challenge" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "challenge2023.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "challenge2023.zip").replace(
                                            ' ', '%20').replace('\\', '/'),
                                        # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/OP7fl7h25NqGfcN # user for tracking splits
                                        user="OP7fl7h25NqGfcN",
                                        password=password,
                                        verbose=verbose)
                if res == 1:  # HTTPError
                    print(
                        "This function should download challenge.zip for tracking - but challenge.zip was not uploaded on the server yet!")

            if "test_labels" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "test2023-private.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "test2023-private.zip").replace(
                                            ' ', '%20').replace('\\', '/'),
                                        # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/PvYoZG1INJEp7fk # user for tracking splits GT
                                        user="PvYoZG1INJEp7fk",
                                        password=password,
                                        verbose=verbose)
                if res == 1:  # HTTPError
                    print("This function should download test_labels.zip for tracking - but test_labels.zip was not uploaded on the server yet! - or check the password")

            if "challenge_labels" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "challenge2023-private.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "challenge2023-private.zip").replace(
                                            ' ', '%20').replace('\\', '/'),
                                        # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/PvYoZG1INJEp7fk # user for tracking splits GT
                                        user="PvYoZG1INJEp7fk",
                                        password=password,
                                        verbose=verbose)
                if res == 1:  # HTTPError
                    print("This function should download challenge_labels.zip for tracking - but challenge_labels.zip was not uploaded on the server yet! - or check the password")
        elif task == "jersey-2023":
            if "train" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "train.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "train.zip").replace(
                                            ' ', '%20').replace('\\', '/'),
                                        # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/ejGBsiDr47cTXnf # user for jersey splits
                                        user="ejGBsiDr47cTXnf",
                                        password=password,
                                        verbose=verbose)
                if res == 1:  # HTTPError
                    print(
                        "This function should download train.zip for jersey - but train.zip was not uploaded on the server yet!")

            if "test" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "test.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "test.zip").replace(
                                            ' ', '%20').replace('\\', '/'),
                                        # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/ejGBsiDr47cTXnf # user for jersey splits
                                        user="ejGBsiDr47cTXnf",
                                        password=password,
                                        verbose=verbose)
                if res == 1:  # HTTPError
                    print(
                        "This function should download test.zip for jersey - but test.zip was not uploaded on the server yet!")

            if "challenge" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "challenge.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "challenge.zip").replace(
                                            ' ', '%20').replace('\\', '/'),
                                        # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/ejGBsiDr47cTXnf # user for jersey splits
                                        user="ejGBsiDr47cTXnf",
                                        password=password,
                                        verbose=verbose)
                if res == 1:  # HTTPError
                    print(
                        "This function should download challenge.zip for jersey - but challenge.zip was not uploaded on the server yet!")

            if "test_labels" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "test_gt.json"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "test_gt.json").replace(
                                            ' ', '%20').replace('\\', '/'),
                                        # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/KQ8VcRcKTSbYx1S # user for jersey splits GT
                                        user="KQ8VcRcKTSbYx1S",
                                        password=password,
                                        verbose=verbose)
                if res == 1:  # HTTPError
                    print("This function should download test_labels.zip for jersey - but test_labels.zip was not uploaded on the server yet! - or check the password")

            if "challenge_labels" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "challenge_gt.json"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "challenge_gt.json").replace(
                                            ' ', '%20').replace('\\', '/'),
                                        # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/KQ8VcRcKTSbYx1S # user for jersey splits GT
                                        user="KQ8VcRcKTSbYx1S",
                                        password=password,
                                        verbose=verbose)
                if res == 1:  # HTTPError
                    print("This function should download challenge_gt.json for jersey - but challenge_labels.zip was not uploaded on the server yet! - or check the password")
        elif task == "spotting-2023":
            if "train" in split:
                self.LocalDirectory = os.path.join(self.LocalDirectory, task)
                self.downloadGames(files=["1_baidu_soccer_embeddings.npy","2_baidu_soccer_embeddings.npy", "Labels-v2.json"], split="train")
                self.LocalDirectory = os.path.dirname(self.LocalDirectory)
            
            if "valid" in split:
                self.LocalDirectory = os.path.join(self.LocalDirectory, task)
                self.downloadGames(files=["1_baidu_soccer_embeddings.npy","2_baidu_soccer_embeddings.npy", "Labels-v2.json"], split="valid")
                self.LocalDirectory = os.path.dirname(self.LocalDirectory)

            if "test" in split:
                self.LocalDirectory = os.path.join(self.LocalDirectory, task)
                self.downloadGames(files=["1_baidu_soccer_embeddings.npy","2_baidu_soccer_embeddings.npy", "Labels-v2.json"], split="test")
                self.LocalDirectory = os.path.dirname(self.LocalDirectory)

            if "challenge" in split:
                self.LocalDirectory = os.path.join(self.LocalDirectory, task)
                self.downloadGames(files=["1_baidu_soccer_embeddings.npy","2_baidu_soccer_embeddings.npy"], split="challenge")
                self.LocalDirectory = os.path.dirname(self.LocalDirectory)

            if "test_labels" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "test_labels.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "test_labels.zip").replace(
                                            ' ', '%20').replace('\\', '/'),
                                        # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/8N4bUNJwlEh2rWL # user for spotting splits GT
                                        user="8N4bUNJwlEh2rWL",
                                        password=password,
                                        verbose=verbose)

            if "challenge_labels" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "challenge_labels.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "challenge_labels.zip").replace(
                                            ' ', '%20').replace('\\', '/'),
                                        # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/8N4bUNJwlEh2rWL # user for spotting splits GT
                                        user="8N4bUNJwlEh2rWL",
                                        password=password,
                                        verbose=verbose)
        elif task == "caption-2023":
            if "train" in split:
                self.LocalDirectory = os.path.join(self.LocalDirectory, task)
                self.downloadGames(files=["1_baidu_soccer_embeddings.npy",
                                          "2_baidu_soccer_embeddings.npy", "Labels-caption.json"], split="train", task="caption")
                self.LocalDirectory = os.path.dirname(self.LocalDirectory)

            if "valid" in split:
                self.LocalDirectory = os.path.join(self.LocalDirectory, task)
                self.downloadGames(files=["1_baidu_soccer_embeddings.npy",
                                          "2_baidu_soccer_embeddings.npy", "Labels-caption.json"], split="valid", task="caption")
                self.LocalDirectory = os.path.dirname(self.LocalDirectory)

            if "test" in split:
                self.LocalDirectory = os.path.join(self.LocalDirectory, task)
                self.downloadGames(files=["1_baidu_soccer_embeddings.npy",
                                          "2_baidu_soccer_embeddings.npy", "Labels-caption.json"], split="test", task="caption")
                self.LocalDirectory = os.path.dirname(self.LocalDirectory)

            if "challenge" in split:
                self.LocalDirectory = os.path.join(self.LocalDirectory, task)
                self.downloadGames(files=[
                                   "1_baidu_soccer_embeddings.npy", "2_baidu_soccer_embeddings.npy"], split="challenge", task="caption")
                self.LocalDirectory = os.path.dirname(self.LocalDirectory)

            if "test_labels" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "test_labels.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "test_labels.zip").replace(
                                            ' ', '%20').replace('\\', '/'),
                                        # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/8eaHb2gyDsEqTlI # user for caption splits GT
                                        user="8eaHb2gyDsEqTlI",
                                        password=password,
                                        verbose=verbose)

            if "challenge_labels" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "challenge_labels.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "challenge_labels.zip").replace(
                                            ' ', '%20').replace('\\', '/'),
                                        # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/8eaHb2gyDsEqTlI # user for caption splits GT
                                        user="8eaHb2gyDsEqTlI",
                                        password=password,
                                        verbose=verbose)


        # 2022 and Before
        elif task == "calibration":
            if "train" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "train.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "train.zip").replace(' ', '%20').replace('\\', '/'),
                                        user="UxCNUvxClQ8Hw2R",  # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/UxCNUvxClQ8Hw2R # user for calibration splits
                                        password=password,
                                        verbose=verbose)
            if "valid" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "valid.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "valid.zip").replace(' ', '%20').replace('\\', '/'),
                                        user="UxCNUvxClQ8Hw2R",  # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/UxCNUvxClQ8Hw2R # user for calibration splits
                                        password=password,
                                        verbose=verbose)
            if "test" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "test.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "test.zip").replace(' ', '%20').replace('\\', '/'),
                                        user="UxCNUvxClQ8Hw2R",  # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/UxCNUvxClQ8Hw2R # user for calibration splits
                                        password=password,
                                        verbose=verbose)
            if "challenge" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "challenge.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "challenge.zip").replace(' ', '%20').replace('\\', '/'),
                                        user="UxCNUvxClQ8Hw2R",  # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/UxCNUvxClQ8Hw2R # user for calibration splits
                                        password=password,
                                        verbose=verbose)
            if "test_labels" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "calibration_test_json.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "calibration_test_json.zip").replace(' ', '%20').replace('\\', '/'),
                                        user="n8J8hetGNT43KLX",  # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/n8J8hetGNT43KLX # user for calibration splits GT
                                        password=password,
                                        verbose=verbose)
            if "challenge_labels" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "calibration_challenge_json.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "calibration_challenge_json.zip").replace(' ', '%20').replace('\\', '/'),
                                        user="n8J8hetGNT43KLX",  # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/n8J8hetGNT43KLX # user for calibration splits GT
                                        password=password,
                                        verbose=verbose)

        # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/Ffr8fsJcljh2Ds5 # user for reid splits
        # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/H16Jx8AD39RzhFU # user for reid splits GT 
        elif task == "reid":
            if "train" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "train.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "train.zip").replace(' ', '%20').replace('\\', '/'),
                                        user="Ffr8fsJcljh2Ds5",  # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/Ffr8fsJcljh2Ds5 # user for reid splits
                                        password=password,
                                        verbose=verbose)
                if res == 1: #HTTPError
                    print("This function should download train.zip for reid - but train.zip was not uploaded on the server yet!")
            
            if "valid" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "valid.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "valid.zip").replace(' ', '%20').replace('\\', '/'),
                                        user="Ffr8fsJcljh2Ds5",  # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/Ffr8fsJcljh2Ds5 # user for reid splits
                                        password=password,
                                        verbose=verbose)
                if res == 1: #HTTPError
                    print("This function should download valid.zip for reid - but valid.zip was not uploaded on the server yet!")
            
            if "test" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "test.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "test.zip").replace(' ', '%20').replace('\\', '/'),
                                        user="Ffr8fsJcljh2Ds5",  # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/Ffr8fsJcljh2Ds5 # user for reid splits
                                        password=password,
                                        verbose=verbose)
                if res == 1: #HTTPError
                    print("This function should download test.zip for reid - but test.zip was not uploaded on the server yet!")
            
            if "challenge" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "challenge.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "challenge.zip").replace(' ', '%20').replace('\\', '/'),
                                        user="Ffr8fsJcljh2Ds5",  # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/Ffr8fsJcljh2Ds5 # user for reid splits
                                        password=password,
                                        verbose=verbose)
                if res == 1: #HTTPError
                    print("This function should download challenge.zip for reid - but challenge.zip was not uploaded on the server yet!")

            if "test_labels" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "test_bbox_info.json"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "test_bbox_info.json").replace(' ', '%20').replace('\\', '/'),
                                        user="H16Jx8AD39RzhFU",  # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/Ffr8fsJcljh2Ds5 # user for reid splits
                                        password=password,
                                        verbose=verbose)
                if res == 1: #HTTPError
                    print("This function should download test_labels.zip for reid - but test_labels.zip was not uploaded on the server yet! - or check the password")
            
            if "challenge_labels" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "challenge_bbox_info.json"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "challenge_bbox_info.json").replace(' ', '%20').replace('\\', '/'),
                                        user="H16Jx8AD39RzhFU",  # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/Ffr8fsJcljh2Ds5 # user for reid splits
                                        password=password,
                                        verbose=verbose)
                if res == 1: #HTTPError
                    print("This function should download challenge_labels.zip for reid - but challenge_labels.zip was not uploaded on the server yet! - or check the password")

        # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/o9tzUs2GcuEwcnr # user for tracking splits
        # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/qWNjAzjEI6hezNf # user for tracking splits GT 
        elif task == "tracking":
            if "train" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "train.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "train.zip").replace(' ', '%20').replace('\\', '/'),
                                        user="o9tzUs2GcuEwcnr",  # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/o9tzUs2GcuEwcnr # user for tracking splits
                                        password=password,
                                        verbose=verbose)
                if res == 1: #HTTPError
                    print("This function should download train.zip for tracking - but train.zip was not uploaded on the server yet!")
                
            if "test" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "test.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "test.zip").replace(' ', '%20').replace('\\', '/'),
                                        user="o9tzUs2GcuEwcnr",  # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/o9tzUs2GcuEwcnr # user for tracking splits
                                        password=password,
                                        verbose=verbose)
                if res == 1: #HTTPError
                    print("This function should download test.zip for tracking - but test.zip was not uploaded on the server yet!")

            if "challenge" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "challenge.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "challenge.zip").replace(' ', '%20').replace('\\', '/'),
                                        user="o9tzUs2GcuEwcnr",  # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/o9tzUs2GcuEwcnr # user for tracking splits
                                        password=password,
                                        verbose=verbose)
                if res == 1: #HTTPError
                    print("This function should download challenge.zip for tracking - but challenge.zip was not uploaded on the server yet!")

            if "test_labels" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "test_labels.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "test_labels.zip").replace(' ', '%20').replace('\\', '/'),
                                        user="qWNjAzjEI6hezNf",  # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/qWNjAzjEI6hezNf # user for tracking splits GT 
                                        password=password,
                                        verbose=verbose)
                if res == 1: #HTTPError
                    print("This function should download test_labels.zip for tracking - but test_labels.zip was not uploaded on the server yet! - or check the password")

            if "challenge_labels" in split:
                res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, task, "challenge_labels.zip"),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "challenge_labels.zip").replace(' ', '%20').replace('\\', '/'),
                                        user="qWNjAzjEI6hezNf",  # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/qWNjAzjEI6hezNf # user for tracking splits GT 
                                        password=password,
                                        verbose=verbose)
                if res == 1: #HTTPError
                    print("This function should download challenge_labels.zip for tracking - but challenge_labels.zip was not uploaded on the server yet! - or check the password")

        elif task == "spotting":
            # When downloading with this function, the data is downloaded on the subfolder "spotting"
            self.LocalDirectory = os.path.join(self.LocalDirectory, "spotting")
            self.downloadGames(files=["1_baidu_soccer_embeddings.npy", "2_baidu_soccer_embeddings.npy", "Labels-v2.json"], split=split)
            self.LocalDirectory = os.path.dirname(self.LocalDirectory)

        else:
            print("ERROR Unknown task:", task)


    def downloadVideoHD(self, game, file):

        FileLocal = os.path.join(self.LocalDirectory, game, file)
        FileURL = os.path.join(self.OwnCloudServer, game, file).replace(' ', '%20')
        FileURL = FileURL.replace('\\', '/')
        if game in getListGames("v1"):
            user = "B72R7dTu1tZtIst"
        if game in getListGames("challenge"):
            user = "gJ8gja7V8SLxYBh"
        res = self.downloadFile(path_local=FileLocal,
                                path_owncloud=FileURL,
                                user=user,  # user for video HQ
                                password=self.password)


    def downloadVideo(self, game, file):

        FileLocal = os.path.join(self.LocalDirectory, game, file)
        FileURL = os.path.join(self.OwnCloudServer, game, file).replace(' ', '%20')
        FileURL = FileURL.replace('\\', '/')

        if game in getListGames("v1"):
            user = "6XYClm33IyBkTgl"
        if game in getListGames("challenge"):
            user = "trXNXsW9W04onBh"
        res = self.downloadFile(path_local=FileLocal,
                                path_owncloud=FileURL,
                                user=user,  # user for video
                                password=self.password)
                                    
    def downloadGameIndex(self, index, files=["1.mkv", "2.mkv", "Labels.json"], verbose=True):
        return self.downloadGame(getListGames("all")[index], files=files, verbose=verbose)

    def downloadGame(self, game, files=["1.mkv", "2.mkv", "Labels.json"], spl="train", verbose=True):


        for file in files:

            GameDirectory = os.path.join(self.LocalDirectory, game)
            FileURL = os.path.join(self.OwnCloudServer, game, file).replace(' ', '%20')
            FileURL = FileURL.replace('\\', '/')

            os.makedirs(GameDirectory, exist_ok=True)

            # 224p Videos
            if file in ["1_224p.mkv", "2_224p.mkv"]:
                res = self.downloadFile(path_local=os.path.join(GameDirectory, file),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "video_224p", game, file).replace(' ', '%20').replace('\\', '/'),
                                        user="MKmZigARdGSoaTT",  # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/MKmZigARdGSoaTT # user for video 224p
                                        password=self.password,
                                        verbose=verbose)

            # 720p Videos
            if file in ["1_720p.mkv", "2_720p.mkv"]:
                res = self.downloadFile(path_local=os.path.join(GameDirectory, file),
                                        path_owncloud=os.path.join(self.OwnCloudServer, "video_720p", game, file).replace(' ', '%20').replace('\\', '/'),
                                        user="xNGfp1W3wPeVOmQ",  # https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/xNGfp1W3wPeVOmQ # user for video 720p
                                        password=self.password,
                                        verbose=verbose)
            # print(spl)
            if spl == "challenge":  # specific buckets for the challenge set

                # LQ Videos
                if file in ["1.mkv", "2.mkv"]:
                    res = self.downloadFile(path_local=os.path.join(GameDirectory, file),
                                            path_owncloud=FileURL,
                                            user="trXNXsW9W04onBh",  # user for video LQ
                                            password=self.password,
                                            verbose=verbose)

                # HQ Videos
                elif file in ["1_HQ.mkv", "2_HQ.mkv", "video.ini"]:
                    res = self.downloadFile(path_local=os.path.join(GameDirectory, file),
                                            path_owncloud=FileURL,
                                            user="gJ8gja7V8SLxYBh",  # user for video HQ
                                            password=self.password,
                                            verbose=verbose)
                
                # V3
                elif file in ["Labels-v3.json"]:
                    res = self.downloadFile(path_local=os.path.join(GameDirectory, file),
                                            path_owncloud=FileURL,
                                            user="NqU4604el8hssGx",  # shared folder for V3
                                            password=self.password,
                                            verbose=verbose)

                elif file in ["Frames-v3.zip"]:
                    res = self.downloadFile(path_local=os.path.join(GameDirectory, file),
                                            path_owncloud=FileURL,
                                            user="okteXlk6jmDXNJc",  # shared folder for V3
                                            password="SoccerNet_Reviewers_SDATA",
                                            verbose=verbose)

                # Labels
                elif "Labels" in file:
                    # file in ["Labels.json", "Labels_v2.json"]:
                    # elif any(feat in file for feat in ["ResNET", "C3D", "I3D", "R25D"]):
                    res = self.downloadFile(path_local=os.path.join(GameDirectory, file),
                                            path_owncloud=FileURL,
                                            user="WUOSnPSYRC1RY13",  # user for Labels
                                            password=self.password,
                                            verbose=verbose)

                # Features
                elif any(feat in file for feat in ["ResNET", "C3D", "I3D", "R25D", "calibration", "player", "field", "boundingbox", ".npy"]):
                    res = self.downloadFile(path_local=os.path.join(GameDirectory, file),
                                            path_owncloud=FileURL,
                                            user="d4nu5rJ6IilF9B0",  # user for Features
                                            password="SoccerNet",
                                            verbose=verbose)


            else:  # bucket for "v1"
                # LQ Videos
                if file in ["1.mkv", "2.mkv"]:
                    res = self.downloadFile(path_local=os.path.join(GameDirectory, file),
                                            path_owncloud=FileURL,
                                            user="6XYClm33IyBkTgl",  # user for video LQ
                                            password=self.password,
                                            verbose=verbose)

                # HQ Videos
                elif file in ["1_HQ.mkv", "2_HQ.mkv", "video.ini"]:
                    res = self.downloadFile(path_local=os.path.join(GameDirectory, file),
                                            path_owncloud=FileURL,
                                            user="B72R7dTu1tZtIst",  # user for video HQ
                                            password=self.password,
                                            verbose=verbose)

                # V3
                elif file in ["Frames-v3.zip", "Labels-v3.json"]:
                    res = self.downloadFile(path_local=os.path.join(GameDirectory, file),
                                            path_owncloud=FileURL,
                                            user="okteXlk6jmDXNJc",  # shared folder for V3
                                            password="SoccerNet_Reviewers_SDATA", 
                                            verbose=verbose)
                                            
                # Labels
                elif "Labels" in file:
                    # elif file in ["Labels.json"]:
                    res = self.downloadFile(path_local=os.path.join(GameDirectory, file),
                                            path_owncloud=FileURL,
                                            user="ZDeEfBzCzseRCLA",  # user for Labels
                                            password="SoccerNet",
                                            verbose=verbose)

                # features
                elif any(feat in file for feat in ["ResNET", "C3D", "I3D", "R25D", "calibration", "player", "field", "boundingbox", ".npy"]):
                    res = self.downloadFile(path_local=os.path.join(GameDirectory, file),
                                            path_owncloud=FileURL,
                                            user="9eRjic29XTk0gS9",  # user for Features
                                            password="SoccerNet",
                                            verbose=verbose)


    def downloadGames(self, files=["1.mkv", "2.mkv", "Labels.json"], split=["train", "valid", "test"], task="spotting", verbose=True, randomized=False):

        if not isinstance(split, list):
            split = [split]
        for spl in split:

            gamelist = getListGames(spl, task)
            if randomized:
                gamelist = random.sample(gamelist,len(gamelist))

            for game in gamelist:
                self.downloadGame(game=game, files=files, spl=spl, verbose=verbose)



                    
if __name__ == "__main__":

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    # Load the arguments
    parser = ArgumentParser(description='Test Downloader',
                            formatter_class=ArgumentDefaultsHelpFormatter) 

    parser.add_argument('--SoccerNet_path',   required=True,
                        type=str, help='Path to the SoccerNet-V2 dataset folder')
    parser.add_argument('--password',   required=False,
                        type=str, help='Path to the list of games to treat')
    args = parser.parse_args()

    mySoccerNetDownloader = SoccerNetDownloader(args.SoccerNet_path)
    mySoccerNetDownloader.password = args.password
    mySoccerNetDownloader.downloadGameIndex(index=549, files=[
                                       "1_HQ.mkv", "2_HQ.mkv", "video.ini", "Labels.json"])
   