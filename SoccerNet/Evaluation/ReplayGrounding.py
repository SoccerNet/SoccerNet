
from SoccerNet.utils import getListGames
import numpy as np
# from config.classes import  EVENT_DICTIONARY_V2
import json
import os
# import torch
from tqdm import tqdm

from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V2

from SoccerNet.Evaluation.utils import LoadJsonFromZip

import zipfile


import glob

def evaluate(SoccerNet_path, Predictions_path, prediction_file="Detection-replays.json", split="test", framerate=None, metric="loose"):
    framerate = 2
    gt_list, detections_list = replay_from_json(
        path=SoccerNet_path, det_path=Predictions_path, split=split, framerate=framerate, detection_name=prediction_file)
    # if split!="challenge":
    
    if metric == "loose":
        deltas=np.arange(12)*5 + 5
    elif metric == "tight":
        deltas=np.arange(5)*1 + 1
    a_AP = average_mAP(gt_list, detections_list, framerate=framerate,deltas=deltas)
    # print("a_mAP: ",a_mAP)
    return {"a_AP": a_AP}
    # return {"a_AP": -1}

def game_results_to_json(path,split,detections_half1,detections_half2,replay_names__half1,replay_names__half2,framerate,half1_time,half2_time):
    #calculate the json file of oputs for a given game 


    # Get the game game 
    game_list=getListGames(split)
    game=game_list[int(replay_names__half1[1][0][0])]

    #construct the Dict for json file
    data={}
    data["Game:"]=game
    #print("replay_names__half1",len(detections_half1[0]))
    # 
    data["half1_time"]=int(half1_time)//framerate
    data["half2_time"]=int(half2_time)//framerate
    data["Replays"]=[]
    for detection_half1,replay_name__half1 in zip(detections_half1,replay_names__half1):
        replay={}
        replay["half"]=1
        #print("replay_name__half1",replay_name__half1[0,1]//framerate)
        replay["start"]=(replay_name__half1[0,1]//framerate).item()
        replay["end"]=(replay_name__half1[0,2]//framerate).item()
        replay["detection"]=[]
        for i in np.arange(len(detection_half1)):
            if detection_half1[i]!=-1:
                replay["detection"].append({
                'time': int(i)/framerate,
                'score': detection_half1[i,0].item()
                })

        data["Replays"].append(replay)
    half1_time=len(detection_half1)
    #print(half1_time)
    for detection_half2,replay_name__half2 in zip(detections_half2,replay_names__half2):
        replay={}
        replay["half"]=2
        replay["start"]=(replay_name__half2[0,1]//framerate).item()
        replay["end"]=(replay_name__half2[0,2]//framerate).item()
        replay["detection"]=[]
        for i in np.arange(len(detection_half2)):
            if detection_half2[i]!=-1:
                replay["detection"].append({
                'time': int(i)/framerate,
                'score': detection_half2[i,0].item()
                })

        data["Replays"].append(replay)
    half2_time=len(detection_half2)
    detection_name="Detection-replays.json"
    os.makedirs(os.path.join(path, game), exist_ok=True)
    with open(os.path.join(path, game, detection_name), 'w') as outfile:
        json.dump(data, outfile,indent=4)


def replay_from_json(path, det_path, split, framerate, detection_name="Detection-replays.json"):
    # This function gets the paths for detected results and ground thruth and produces the list of vectors for final Evaluation  
    # print(split, framerate)
    game_list=getListGames(split)
    dict_type=EVENT_DICTIONARY_V2
    # detection_name="Detection-replays.json"
    labels_name="Labels-cameras.json"
    # if split=="challenge":
    #     labels_name="Labels-replays.json"
    detections_list=list()
    gt_list=list()
    game_time=np.zeros((2,1))
    for game in tqdm(game_list):
        # Read labels
        # labels_replays = json.load(open(os.path.join(path, game, labels_name)))
        # print(path)
        if zipfile.is_zipfile(path): # deal with zipped folders
            labels_replays = LoadJsonFromZip(path, os.path.join(game, labels_name))
        else:
            labels_replays = json.load(open(os.path.join(path, game, labels_name)))

        # Read detection
        # detection_replays = json.load(open(os.path.join(det_path, game, detection_name)))
        # print(det_path)

        # infer name of the prediction_file
        if detection_name == None:
            if zipfile.is_zipfile(det_path):
                with zipfile.ZipFile(det_path, "r") as z:
                    for filename in z.namelist():
                        #       print(filename)
                        if filename.endswith(".json"):
                            detection_name = os.path.basename(filename)
                            break
            else:
                for filename in glob.glob(os.path.join(det_path,"*/*/*/*.json")):
                    detection_name = os.path.basename(filename)
                    # print(detection_name)
                    break
        # print(detection_name)

        if zipfile.is_zipfile(det_path): # deal with zipped folders
            detection_replays = LoadJsonFromZip(det_path, os.path.join(game, detection_name))
        else:
            detection_replays = json.load(open(os.path.join(det_path, game, detection_name)))


        for replay in detection_replays["Replays"]:
            half=int(replay["half"])
            #print(half)
            start=int(replay["start"])*framerate
            end=int(replay["end"])*framerate
            #print('start-end',start,end)
            #replay_det=np.zeros((int(game_time[half-1,0]),1))
            replay_det=np.zeros([3500*framerate,1], dtype = np.float)-1
            replay_gt=np.zeros([3500*framerate,1], dtype = np.float)
            previous_frame=0
            #print('size',replay_det.shape,replay_gt.shape)
            for detection in replay["detection"]:
                replay_det[int(detection['time']*framerate),0]=detection['score']
            #detections_list.append(torch.from_numpy(replay_det).cpu().detach())
            #detections_list.append(replay_det)
            
            for annotation in labels_replays["annotations"]:
                time = annotation["gameTime"]
                half_label = int(time[0])

                frame=time_to_frame(time,framerate)
                if not "link" in annotation:
                    previous_timestamp = frame
                    continue
                event = annotation["link"]["label"]
                if  int(annotation["link"]["half"]) != half or not event in dict_type :
                    previous_timestamp = frame
                    continue
                if previous_timestamp == frame:
                    previous_timestamp = frame
                    continue
                if frame==end:
                    #print(previous_timestamp,frame,start,end)
                    #print(annotation["link"]["time"])
                    time_event = annotation["link"]["time"]
                    minutes_event = int(time_event[0:2])
                    seconds_event = int(time_event[3:])
                    frame_event = framerate * ( seconds_event + 60 * minutes_event )
                    #link_time=time_to_frame(annotation["link"]["time"],framerate)
                    replay_gt[frame_event]=1
                    #gt_list.append(torch.from_numpy(replay_gt))
                    #detections_list.append(torch.from_numpy(replay_det))														
                    gt_list.append(replay_gt)
                    detections_list.append(replay_det)
                    break			
    return gt_list,detections_list			

def time_to_frame(time,framerate):
    minutes = int(time[-5:-3])
    seconds = int(time[-2::])
    frame = framerate * ( seconds + 60 * minutes ) 
    return frame


# import numpy as np
# from tqdm import tqdm
import time
np.seterr(divide='ignore', invalid='ignore')

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def NMS(detections, delta):
    
    # Array to put the results of the NMS
    detections_tmp = np.copy(detections)
    detections_NMS = np.zeros(detections.shape)-1

    # Return with naive approach
    """
    if detections.shape[0] > 60:
        detections_NMS[-60] = 1
    else:
        detections_NMS[-1] = 1
    return detections_NMS
    """

    # Loop over all classes
    for i in np.arange(detections.shape[-1]):
        # Stopping condition
        while(np.max(detections_tmp[:,i]) >= 0):

            # Get the max remaining index and value
            max_value = np.max(detections_tmp[:,i])
            max_index = np.argmax(detections_tmp[:,i])

            detections_NMS[max_index,i] = max_value

            detections_tmp[int(np.maximum(-(delta/2)+max_index,0)): int(np.minimum(max_index+int(delta/2), detections.shape[0])) ,i] = -1

    return detections_NMS

def compute_class_scores(target, detection, delta):

    # Retrieving the important variables
    gt_indexes = np.where(target != 0)[0]
    pred_indexes = np.where(detection >= 0)[0]
    pred_scores = detection[pred_indexes]

    # Array to save the results, each is [pred_scor,{1 or 0}]
    game_detections = np.zeros((len(pred_indexes),2))
    game_detections[:,0] = np.copy(pred_scores)


    remove_indexes = list()

    for gt_index in gt_indexes:

        max_score = -1
        max_index = None
        game_index = 0
        selected_game_index = 0

        for pred_index, pred_score in zip(pred_indexes, pred_scores):

            if pred_index < gt_index - delta:
                game_index += 1
                continue
            if pred_index > gt_index + delta:
                break
                
            if abs(pred_index-gt_index) <= delta/2 and pred_score > max_score and pred_index not in remove_indexes:
                max_score = pred_score
                max_index = pred_index
                selected_game_index = game_index
            game_index += 1

        if max_index is not None:
            game_detections[selected_game_index,1]=1
            remove_indexes.append(max_index)

    return game_detections, len(gt_indexes)



def compute_precision_recall_curve(targets, detections, delta, NMS_on):
    
    # Store the number of classes
    num_classes = targets[0].shape[-1]

    # 200 confidence thresholds between [0,1]
    thresholds = np.linspace(0,1,200)

    # Store the precision and recall points
    precision = list()
    recall = list()

    # Apply Non-Maxima Suppression if required
    start = time.time()
    detections_NMS = list()
    if NMS_on:
        for detection in detections:
            detections_NMS.append(NMS(detection,delta))
    else:
        detections_NMS = detections

    # Precompute the predictions scores and their correspondence {TP, FP} for each class
    for c in np.arange(num_classes):
        total_detections =  np.zeros((1, 2))
        total_detections[0,0] = -1
        n_gt_labels = 0
        
        # Get the confidence scores and their corresponding TP or FP characteristics for each game
        for target, detection in zip(targets, detections_NMS):
            tmp_detections, tmp_n_gt_labels = compute_class_scores(target[:,c], detection[:,c], delta)
            total_detections = np.append(total_detections,tmp_detections,axis=0)
            n_gt_labels = n_gt_labels + tmp_n_gt_labels

        precision.append(list())
        recall.append(list())

        # Get the precision and recall for each confidence threshold
        for threshold in thresholds:
            pred_indexes = np.where(total_detections[:,0]>=threshold)[0]
            TP = np.sum(total_detections[pred_indexes,1])
            p = np.nan_to_num(TP/len(pred_indexes))
            r = np.nan_to_num(TP/n_gt_labels)
            precision[-1].append(p)
            recall[-1].append(r)

    precision = np.array(precision).transpose()
    recall = np.array(recall).transpose()


    # Sort the points based on the recall, class per class
    for i in np.arange(num_classes):
        index_sort = np.argsort(recall[:,i])
        precision[:,i] = precision[index_sort,i]
        recall[:,i] = recall[index_sort,i]

    return precision, recall

def compute_mAP(precision, recall):

    # Array for storing the AP per class
    AP = np.array([0.0]*precision.shape[-1])

    # Loop for all classes
    for i in np.arange(precision.shape[-1]):

        # 11 point interpolation
        for j in np.arange(11)/10:

            index_recall = np.where(recall[:,i] >= j)[0]

            possible_value_precision = precision[index_recall,i]
            max_value_precision = 0

            if possible_value_precision.shape[0] != 0:
                max_value_precision = np.max(possible_value_precision)

            AP[i] += max_value_precision

    mAP_per_class = AP/11

    return np.mean(mAP_per_class)

def delta_curve(targets, detections,  framerate, NMS_on, deltas=np.arange(5)*1 + 1):

    mAP = list()

    for delta in tqdm(deltas*framerate):

        precision, recall = compute_precision_recall_curve(targets, detections, delta, NMS_on)

        mAP.append(compute_mAP(precision, recall))

    return mAP


def average_mAP(targets, detections, framerate=2, NMS_on=True, deltas=np.arange(5)*1 + 1):

    targets_numpy = list()
    detections_numpy = list()
    
    for target, detection in zip(targets,detections):
        targets_numpy.append(target)
        detections_numpy.append(detection)

    mAP = delta_curve(targets_numpy, detections_numpy, framerate, NMS_on, deltas)
    # Compute the average mAP
    integral = 0.0
    for i in np.arange(len(mAP)-1):
        integral += 5*(mAP[i]+mAP[i+1])/2
    a_mAP = integral/(5*(len(mAP)-1))

    return a_mAP


def average_mAP_heuristic(targets, detections, replays, framerate=2, seconds=0, NMS_on=True, deltas=np.arange(5)*1 + 1):

    targets_numpy = list()
    detections_numpy = list()
    replays_numpy = list()
    
    for target, detection, replay in zip(targets,detections,replays):
        targets_numpy.append(target.numpy())
        replay_numpy = replay.numpy()

        replay_stamp = max(0,int(np.where(replay_numpy)[0][0]-framerate*seconds))
        detection_heuristic = np.zeros(detection.numpy().shape)-1
        detection_heuristic[replay_stamp,0] = 1

        detections_numpy.append(detection_heuristic)

    mAP = delta_curve(targets_numpy, detections_numpy, framerate, NMS_on, deltas)
    # Compute the average mAP
    integral = 0.0
    for i in np.arange(len(mAP)-1):
        integral += 5*(mAP[i]+mAP[i+1])/2
    a_mAP = integral/(5*(len(mAP)-1))

    return a_mAP
