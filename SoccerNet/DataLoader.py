
# from .utils import getListGames
import os

import numpy as np
import cv2  # pip install opencv-python (==3.4.11.41)
import imutils  # pip install imutils
import skvideo.io
from tqdm import tqdm
import logging
import moviepy.editor

def getDuration(video_path):
    """Get the duration (in seconds) for a video.

    Keyword arguments:
    video_path -- the path of the video
    """
    return moviepy.editor.VideoFileClip(video_path).duration

import cv2
class FrameCV():
    def __init__(self, video_path, FPS=2, transform=None, start=None, duration=None):
        """Create a list of frame from a video using OpenCV.

        Keyword arguments:
        video_path -- the path of the video
        FPS -- the desired FPS for the frames (default:2)
        transform -- the desired transformation for the frames (default:2)
        start -- the desired starting time for the list of frames (default:None)
        duration -- the desired duration time for the list of frames (default:None)
        """

        self.FPS = FPS
        self.transform = transform
        self.start = start
        self.duration = duration

        # read video
        vidcap = cv2.VideoCapture(video_path)
        # read FPS
        self.fps_video = vidcap.get(cv2.CAP_PROP_FPS)
        # read duration
        self.time_second = getDuration(video_path)        

        # loop until the number of frame is consistent with the expected number of frame, 
        # given the duratio nand the FPS
        good_number_of_frames = False
        while not good_number_of_frames: 

            # read video
            vidcap = cv2.VideoCapture(video_path)
            
            # get number of frames
            self.numframe = int(self.time_second*self.fps_video)
            
            # frame drop ratio
            drop_extra_frames = self.fps_video/self.FPS

            # init list of frames
            self.frames = []

            # TQDM progress bar
            pbar = tqdm(range(self.numframe), desc='Grabbing Video Frames', unit='frame')
            i_frame = 0
            ret, frame = vidcap.read()

            # loop until no frame anymore
            while ret:
                # update TQDM
                pbar.update(1)
                i_frame += 1
                
                # skip until starting time
                if self.start is not None:
                    if i_frame < self.fps_video * self.start:
                        ret, frame = vidcap.read()
                        continue

                # skip after duration time
                if self.duration is not None:
                    if i_frame > self.fps_video * (self.start + self.duration):
                        ret, frame = vidcap.read()
                        continue
                        

                if (i_frame % drop_extra_frames < 1):

                    # crop keep the central square of the frame
                    if self.transform == "resize256crop224":  
                        frame = imutils.resize(frame, height=256)  # keep aspect ratio
                        # number of pixel to remove per side
                        off_h = int((frame.shape[0] - 224)/2)
                        off_w = int((frame.shape[1] - 224)/2)
                        frame = frame[off_h:-off_h,
                                      off_w:-off_w, :]  # remove pixel at each side

                    # crop remove the side of the frame
                    elif self.transform == "crop":  
                        frame = imutils.resize(frame, height=224)  # keep aspect ratio
                        # number of pixel to remove per side
                        off_side = int((frame.shape[1] - 224)/2)
                        frame = frame[:, off_side:-
                                        off_side, :]  # remove them

                    # resize change the aspect ratio
                    elif self.transform == "resize":  
                        # lose aspect ratio
                        frame = cv2.resize(frame, (224, 224),
                                            interpolation=cv2.INTER_CUBIC)

                    # append the frame to the list
                    self.frames.append(frame)
                
                # read next frame
                ret, frame = vidcap.read()

            # check if the expected number of frames were read
            if self.numframe - (i_frame+1) <=1:
                logging.debug("Video read properly")
                good_number_of_frames = True
            else:
                logging.debug("Video NOT read properly, adjusting fps and read again")
                self.fps_video = (i_frame+1) / self.time_second

        # convert frame from list to numpy array
        self.frames = np.array(self.frames)

    def __len__(self):
        """Return number of frames."""
        return len(self.frames)

    def __iter__(self, index):
        """Return frame at given index."""
        return self.frames[index]
    


class Frame():
    def __init__(self, video_path, FPS=2, transform=None, start=None, duration=None):

        self.FPS = FPS
        self.transform = transform

        # Knowing number of frames from FFMPEG metadata w/o without iterating over all frames
        videodata = skvideo.io.FFmpegReader(video_path)
        # numFrame x H x W x channels
        (numframe, _, _, _) = videodata.getShape()
        # if self.verbose:
            # print("shape video", videodata.getShape())
        self.time_second = getDuration(video_path)
        # fps_video = numframe / time_second

        # time_second = getDuration(video_path)
        # if self.verbose:
        #     print("duration video", time_second)

        good_number_of_frames = False
        while not good_number_of_frames:
            fps_video = numframe / self.time_second
            # time_second = numframe / fps_video

            self.frames = []
            videodata = skvideo.io.vreader(video_path)
            # fps_desired = 2
            drop_extra_frames = fps_video/self.FPS

            # print(int(fps_video * start), int(fps_video * (start+45*60)))
            for i_frame, frame in tqdm(enumerate(videodata), total=numframe):
                # print(i_frame)

                for t in [0,5,10,15,20,25,30,35,40,45]:

                    if start is not None:
                        if i_frame == int(fps_video * (start + t*60)):
                        # print("saving image")
                            skvideo.io.vwrite(video_path.replace(".mkv", f"snap_{t}.png"), frame)
                            # os.path.join(os.path.dirname(video_path), f"snap_{t}.png"), frame)
                    # if i_frame == int(fps_video * (start+45*60)):
                    #     print("saving image")
                    #     skvideo.io.vwrite(os.path.join(os.path.dirname(video_path), "45.png"), frame)

                if start is not None:
                    if i_frame < fps_video * start:
                        continue

                if duration is not None:
                    if i_frame > fps_video * (start + duration):
                        # print("end of duration :)")
                        continue

                if (i_frame % drop_extra_frames < 1):

                    if self.transform == "resize256crop224":  # crop keep the central square of the frame
                        frame = imutils.resize(
                            frame, height=256)  # keep aspect ratio
                        # number of pixel to remove per side
                        off_side_h = int((frame.shape[0] - 224)/2)
                        off_side_w = int((frame.shape[1] - 224)/2)
                        frame = frame[off_side_h:-off_side_h,
                                        off_side_w:-off_side_w, :]  # remove them

                    elif self.transform == "crop":  # crop keep the central square of the frame
                        frame = imutils.resize(
                            frame, height=224)  # keep aspect ratio
                        # number of pixel to remove per side
                        off_side = int((frame.shape[1] - 224)/2)
                        frame = frame[:, off_side:-
                                        off_side, :]  # remove them

                    elif self.transform == "resize":  # resize change the aspect ratio
                        # lose aspect ratio
                        frame = cv2.resize(frame, (224, 224),
                                            interpolation=cv2.INTER_CUBIC)

                    # else:
                    #     raise NotImplmentedError()
                    # if self.array:
                    #     frame = img_to_array(frame)
                    self.frames.append(frame)

            print("expected number of frames", numframe,
                  "real number of available frames", i_frame+1)

            if numframe == i_frame+1:
                print("===>>> proper read! Proceeding! :)")
                good_number_of_frames = True
            else:
                print("===>>> not read properly... Read frames again! :(")
                numframe = i_frame+1
        
        self.frames = np.array(self.frames)

    def __len__(self):
        """Return number of frames."""
        return len(self.frames)

    def __iter__(self, index):
        """Return frame at given index."""
        return self.frames[index]
    
    # def frames(self):

# class VideoLoader():
#     def __init__(self, SoccerNetDir, split="v1"):
#         self.SoccerNetDir = SoccerNetDir
#         self.split = split
#         # if split == "v1":
#         #     self.listGame = getListGames("v1")
#         # # elif split == "challenge":
#         # #     self.listGame = getListGames()
#         # else:
#         self.listGame = getListGames(split)

#     def __len__(self):
#         return len(self.listGame)

#     def __iter__(self, index):
#         video_path = self.listGame[index]

#         # Read RELIABLE lenght for the video, in second
#         if args.verbose:
#             print("video path", video_path)
#         v = cv2.VideoCapture(video_path)
#         v.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
#         time_second = v.get(cv2.CAP_PROP_POS_MSEC)/1000
#         if args.verbose:
#             print("duration video", time_second)
#         import json
#         metadata = skvideo.io.ffprobe(video_path)
#         # print(metadata.keys())
#         # print(json.dumps(metadata["video"], indent=4))
#         # getduration
#         # print(metadata["video"]["@avg_frame_rate"])
#         # # print(metadata["video"]["@duration"])

#         # Knowing number of frames from FFMPEG metadata w/o without iterating over all frames
#         videodata = skvideo.io.FFmpegReader(video_path)
#         (numframe, _, _, _) = videodata.getShape()  # numFrame x H x W x channels
#         if args.verbose:
#             print("shape video", videodata.getShape())

#         # # extract REAL FPS
#         fps_video = metadata["video"]["@avg_frame_rate"]
#         fps_video = float(fps_video.split("/")[0])/float(fps_video.split("/")[1])
#         # fps_video = numframe/time_second
#         if args.verbose:
#             print("fps=", fps_video)
#         time_second = numframe / fps_video
#         if args.verbose:
#             print("duration video", time_second)
#         frames = []
#         videodata = skvideo.io.vreader(video_path)
#         fps_desired = 2
#         drop_extra_frames = fps_video/fps_desired
#         for i_frame, frame in tqdm(enumerate(videodata), total=numframe):
#             # print(i_frame % drop_extra_frames)
#             if (i_frame % drop_extra_frames < 1):

#                 if args.preprocess == "resize256crop224":  # crop keep the central square of the frame
#                     frame = imutils.resize(frame, height=256)  # keep aspect ratio
#                     # number of pixel to remove per side
#                     off_side_h = int((frame.shape[0] - 224)/2)
#                     off_side_w = int((frame.shape[1] - 224)/2)
#                     frame = frame[off_side_h:-off_side_h,
#                                 off_side_w:-off_side_w, :]  # remove them

#                 elif args.preprocess == "crop":  # crop keep the central square of the frame
#                     frame = imutils.resize(frame, height=224)  # keep aspect ratio
#                     # number of pixel to remove per side
#                     off_side = int((frame.shape[1] - 224)/2)
#                     frame = frame[:, off_side:-off_side, :]  # remove them

#                 elif args.preprocess == "resize":  # resize change the aspect ratio
#                     # lose aspect ratio
#                     frame = cv2.resize(frame, (224, 224),
#                                     interpolation=cv2.INTER_CUBIC)

#                 else:
#                     raise NotImplmentedError()
#                 frames.append(frame)

#         # create numpy aray (nb_frames x 224 x 224 x 3)
#         frames = np.array(frames)
#         return frames


if __name__ == "__main__":
    from SoccerNet.utils import getListGames
    import argparse
    

    parser = argparse.ArgumentParser(description='Test dataloader.')

    parser.add_argument('--soccernet_dirpath', type=str, default="/media/giancos/Football/SoccerNet_HQ/",
                        help="Path for SoccerNet directory [default:/media/giancos/Football/SoccerNet_HQ/]")
    parser.add_argument('--idx_game', type=int, default=0,
                        help="index of the game ot test [default:0]")
    args = parser.parse_args()
    print(args)

    # read ini
    import configparser
    config = configparser.ConfigParser()
    config.read(os.path.join(args.soccernet_dirpath, getListGames("all")[args.idx_game], "video.ini"))

    # loop over videos in game
    for vid in config.sections():
        video_path = os.path.join(args.soccernet_dirpath, getListGames("all")[args.idx_game], vid)

        print(video_path)
        loader = FrameCV(video_path, 
            start=float(config[vid]["start_time_second"]), 
            duration=float(config[vid]["duration_second"]))
        print(loader)
