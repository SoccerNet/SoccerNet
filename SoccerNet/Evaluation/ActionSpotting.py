import os
import numpy as np

from SoccerNet.utils import getListGames


import json

from SoccerNet.Evaluation.utils import LoadJsonFromZip, EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V1, INVERSE_EVENT_DICTIONARY_V1, EVENT_DICTIONARY_BALL

import json
import zipfile
from tqdm import tqdm


import glob
import zipfile

def inferListGame(SoccerNet_path):
    if zipfile.is_zipfile(SoccerNet_path):

        zip_folder_path = SoccerNet_path

        with zipfile.ZipFile(zip_folder_path, 'r') as zip_ref:
            list_json = [f for f in zip_ref.namelist() if f.endswith('.json')]
            list_games = [os.path.dirname(f) for f in list_json]

    else:
        list_games = []
        for root, dirs, files in os.walk(SoccerNet_path):
            for file in files:
                if file.endswith(".json"):
                    list_games.append(os.path.relpath(root, SoccerNet_path))

    # for game in list_games:
    #     print(" --- ", game)
    
    return list_games


def evaluate(
        SoccerNet_path, Predictions_path, 
        prediction_file="results_spotting.json", # DEPRECATED, set to None and infer name instead. Kept for backward compatibility
        split="test", 
        version=2, 
        framerate=2, 
        metric="loose", 
        label_files="Labels-v2.json", # DEPRECATED, set to None and infer name instead. Kept for backward compatibility
        num_classes=17, # DEPRECATED, use EVENT_DICTIONARY instead. Kept for backward compatibility
        dataset="SoccerNet", # DEPRECATED, set to None and use inferGameList instead. Kept for backward compatibility
        task="spotting",
        EVENT_DICTIONARY=None, # Set to None to use the default EVENT_DICTIONARY from previous datasets
        ):
    # evaluate the prediction with respect to some ground truth
    # Params:
    #   - SoccerNet_path: path for labels (folder or zipped file)
    #   - Predictions_path: path for predictions (folder or zipped file)
    #   - prediction_file: name of the predicted files - if set to None, try to infer it
    #   - split: split to evaluate from ["test", "challenge"]
    #   - frame_rate: frame rate to evalaute from [2]
    # Return:
    #   - details mAP

    if dataset is None:
        list_games = inferListGame(SoccerNet_path=SoccerNet_path)
    else:
        list_games = getListGames(split=split, dataset=dataset, task=task)
    targets_numpy = list()
    detections_numpy = list()
    closests_numpy = list()

    # Set EVENT_DICTIONARY if None, else use the one provided
    if EVENT_DICTIONARY is None:
        if dataset == "SoccerNet" and version == 1 and task == "spotting":
            EVENT_DICTIONARY = EVENT_DICTIONARY_V1
        elif dataset == "SoccerNet" and version == 2 and task == "spotting":
            EVENT_DICTIONARY = EVENT_DICTIONARY_V2
        elif dataset == "Headers":
            EVENT_DICTIONARY = {"Header": 0}
        elif dataset == "Headers-headimpacttype":
            EVENT_DICTIONARY = {"1. Purposeful header": 0, "2. Header Duel": 1,
                                "3. Attempted header": 2, "4. Unintentional header": 3, "5. Other head impacts": 4}
        elif dataset == "Ball":
            EVENT_DICTIONARY = EVENT_DICTIONARY_BALL
        elif dataset == "SoccerNet" and task == "caption":
            EVENT_DICTIONARY = {"comments": 0}
    

    num_classes = len(EVENT_DICTIONARY)

    for game in tqdm(list_games):

        # # Load labels
        # if version==2:
        #     label_files = "Labels-v2.json"
        #     num_classes = 17
        # elif version==1:
        #     label_files = "Labels.json"
        #     num_classes = 3
        # if dataset == "Headers":
        #     label_files = "Labels-Header.json"
        #     num_classes = 3


        # infer name of the label_files
        if label_files == None:
            if zipfile.is_zipfile(SoccerNet_path):
                with zipfile.ZipFile(SoccerNet_path, "r") as z:
                    for filename in z.namelist():
                        if filename.endswith(".json"):
                            label_files = os.path.basename(filename)
                            break
            else:
                for root, dirs, files in os.walk(SoccerNet_path):
                    for file in files:
                        if file.endswith(".json"):
                            label_files = file
                            break
        
        if zipfile.is_zipfile(SoccerNet_path):
            labels = LoadJsonFromZip(SoccerNet_path, os.path.join(game, label_files))
        else:
            labels = json.load(open(os.path.join(SoccerNet_path, game, label_files)))
        # convert labels to vector
        label_half_1, label_half_2 = label2vector(
            labels, num_classes=num_classes, version=version, EVENT_DICTIONARY=EVENT_DICTIONARY, framerate=framerate)




        # infer name of the prediction_file
        if prediction_file == None:
            if zipfile.is_zipfile(Predictions_path):
                with zipfile.ZipFile(Predictions_path, "r") as z:
                    for filename in z.namelist():
                        #       print(filename)
                        if filename.endswith(".json"):
                            prediction_file = os.path.basename(filename)
                            break
            else:
                for filename in glob.glob(os.path.join(Predictions_path,"*/*/*/*.json")):
                    prediction_file = os.path.basename(filename)
                    # print(prediction_file)
                    break

        # Load predictions
        if zipfile.is_zipfile(Predictions_path):
            predictions = LoadJsonFromZip(Predictions_path, os.path.join(game, prediction_file))
        else:
            predictions = json.load(open(os.path.join(Predictions_path, game, prediction_file)))
        # convert predictions to vector
        predictions_half_1, predictions_half_2 = predictions2vector(
            predictions, num_classes=num_classes, version=version, EVENT_DICTIONARY=EVENT_DICTIONARY, framerate=framerate)

        targets_numpy.append(label_half_1)
        targets_numpy.append(label_half_2)
        detections_numpy.append(predictions_half_1)
        detections_numpy.append(predictions_half_2)

        closest_numpy = np.zeros(label_half_1.shape)-1
        #Get the closest action index
        for c in np.arange(label_half_1.shape[-1]):
            indexes = np.where(label_half_1[:,c] != 0)[0].tolist()
            if len(indexes) == 0 :
                continue
            indexes.insert(0,-indexes[0])
            indexes.append(2*closest_numpy.shape[0])
            for i in np.arange(len(indexes)-2)+1:
                start = max(0,(indexes[i-1]+indexes[i])//2)
                stop = min(closest_numpy.shape[0], (indexes[i]+indexes[i+1])//2)
                closest_numpy[start:stop,c] = label_half_1[indexes[i],c]
        closests_numpy.append(closest_numpy)

        closest_numpy = np.zeros(label_half_2.shape)-1
        for c in np.arange(label_half_2.shape[-1]):
            indexes = np.where(label_half_2[:,c] != 0)[0].tolist()
            if len(indexes) == 0 :
                continue
            indexes.insert(0,-indexes[0])
            indexes.append(2*closest_numpy.shape[0])
            for i in np.arange(len(indexes)-2)+1:
                start = max(0,(indexes[i-1]+indexes[i])//2)
                stop = min(closest_numpy.shape[0], (indexes[i]+indexes[i+1])//2)
                closest_numpy[start:stop,c] = label_half_2[indexes[i],c]
        closests_numpy.append(closest_numpy)


    if metric == "loose":
        deltas=np.arange(12)*5 + 5
    elif metric == "tight":
        deltas=np.arange(5)*1 + 1
    elif metric == "at1":
        deltas=np.array([1]) #np.arange(1)*1 + 1
    elif metric == "at2":
        deltas=np.array([2]) 
    elif metric == "at3":
        deltas=np.array([3]) 
    elif metric == "at4":
        deltas=np.array([4]) 
    elif metric == "at5":
        deltas=np.array([5]) 
    # Compute the performances
    a_mAP, a_mAP_per_class, a_mAP_visible, a_mAP_per_class_visible, a_mAP_unshown, a_mAP_per_class_unshown = average_mAP(targets_numpy, detections_numpy, closests_numpy, framerate, deltas=deltas)
    
    results = {
        "a_mAP": a_mAP,
        "a_mAP_per_class": a_mAP_per_class,
        "a_mAP_visible": a_mAP_visible if version==2 else None,
        "a_mAP_per_class_visible": a_mAP_per_class_visible if version==2 else None,
        "a_mAP_unshown": a_mAP_unshown if version==2 else None,
        "a_mAP_per_class_unshown": a_mAP_per_class_unshown if version==2 else None,
    }
    return results


def label2vector(labels, num_classes=17, framerate=2, version=2, EVENT_DICTIONARY={}):


    vector_size = 120*60*framerate

    label_half1 = np.zeros((vector_size, num_classes))
    label_half2 = np.zeros((vector_size, num_classes))

    for annotation in labels["annotations"]:

        time = annotation["gameTime"]
        event = annotation["label"]

        half = int(time[0])

        minutes = int(time[-5:-3])
        seconds = int(time[-2::])
        # annotation at millisecond precision
        if "position" in annotation:
            frame = int(framerate * ( int(annotation["position"])/1000 )) 
        # annotation at second precision
        else:
            frame = framerate * ( seconds + 60 * minutes ) 

        if version == 2:
            if event not in EVENT_DICTIONARY:
                continue
            label = EVENT_DICTIONARY[event]
        elif version == 1:
            # print(event)
            # label = EVENT_DICTIONARY[event]
            if "card" in event: label = 0
            elif "subs" in event: label = 1
            elif "soccer" in event: label = 2
            else: 
                # print(event)
                continue
        # print(event, label, half)

        value = 1
        if "visibility" in annotation.keys():
            if annotation["visibility"] == "not shown":
                value = -1

        if half == 1:
            frame = min(frame, vector_size-1)
            label_half1[frame][label] = value

        if half == 2:
            frame = min(frame, vector_size-1)
            label_half2[frame][label] = value

    return label_half1, label_half2

def predictions2vector(predictions, num_classes=17, version=2, framerate=2, EVENT_DICTIONARY={}):


    vector_size = 120*60*framerate

    prediction_half1 = np.zeros((vector_size, num_classes))-1
    prediction_half2 = np.zeros((vector_size, num_classes))-1

    for annotation in predictions["predictions"]:

        time = int(annotation["position"])
        event = annotation["label"]

        half = int(annotation["half"])

        frame = int(framerate * ( time/1000 ))

        if version == 2:
            if event not in EVENT_DICTIONARY:
                continue
            label = EVENT_DICTIONARY[event]
        elif version == 1:
            label = EVENT_DICTIONARY[event]
            # print(label)
            # EVENT_DICTIONARY_V1[l]
            # if "card" in event: label=0
            # elif "subs" in event: label=1
            # elif "soccer" in event: label=2
            # else: continue

        value = annotation["confidence"]

        if half == 1:
            frame = min(frame, vector_size-1)
            prediction_half1[frame][label] = value

        if half == 2:
            frame = min(frame, vector_size-1)
            prediction_half2[frame][label] = value

    return prediction_half1, prediction_half2


import numpy as np
from tqdm import tqdm
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

def compute_class_scores(target, closest, detection, delta):

    # Retrieving the important variables
    gt_indexes = np.where(target != 0)[0]
    gt_indexes_visible = np.where(target > 0)[0]
    gt_indexes_unshown = np.where(target < 0)[0]
    pred_indexes = np.where(detection >= 0)[0]
    pred_scores = detection[pred_indexes]

    # Array to save the results, each is [pred_scor,{1 or 0}]
    game_detections = np.zeros((len(pred_indexes),3))
    game_detections[:,0] = np.copy(pred_scores)
    game_detections[:,2] = np.copy(closest[pred_indexes])


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

    return game_detections, len(gt_indexes_visible), len(gt_indexes_unshown)



def compute_precision_recall_curve(targets, closests, detections, delta):
    
    # Store the number of classes
    num_classes = targets[0].shape[-1]

    # 200 confidence thresholds between [0,1]
    thresholds = np.linspace(0,1,200)

    # Store the precision and recall points
    precision = list()
    recall = list()
    precision_visible = list()
    recall_visible = list()
    precision_unshown = list()
    recall_unshown = list()

    # Apply Non-Maxima Suppression if required
    start = time.time()

    # Precompute the predictions scores and their correspondence {TP, FP} for each class
    for c in np.arange(num_classes):
        total_detections =  np.zeros((1, 3))
        total_detections[0,0] = -1
        n_gt_labels_visible = 0
        n_gt_labels_unshown = 0
        
        # Get the confidence scores and their corresponding TP or FP characteristics for each game
        for target, closest, detection in zip(targets, closests, detections):
            tmp_detections, tmp_n_gt_labels_visible, tmp_n_gt_labels_unshown = compute_class_scores(target[:,c], closest[:,c], detection[:,c], delta)
            total_detections = np.append(total_detections,tmp_detections,axis=0)
            n_gt_labels_visible = n_gt_labels_visible + tmp_n_gt_labels_visible
            n_gt_labels_unshown = n_gt_labels_unshown + tmp_n_gt_labels_unshown

        precision.append(list())
        recall.append(list())
        precision_visible.append(list())
        recall_visible.append(list())
        precision_unshown.append(list())
        recall_unshown.append(list())

        # Get only the visible or unshown actions
        total_detections_visible = np.copy(total_detections)
        total_detections_unshown = np.copy(total_detections)
        total_detections_visible[np.where(total_detections_visible[:,2] <= 0.5)[0],0] = -1
        total_detections_unshown[np.where(total_detections_unshown[:,2] >= -0.5)[0],0] = -1

        # Get the precision and recall for each confidence threshold
        for threshold in thresholds:
            pred_indexes = np.where(total_detections[:,0]>=threshold)[0]
            pred_indexes_visible = np.where(total_detections_visible[:,0]>=threshold)[0]
            pred_indexes_unshown = np.where(total_detections_unshown[:,0]>=threshold)[0]
            TP = np.sum(total_detections[pred_indexes,1])
            TP_visible = np.sum(total_detections[pred_indexes_visible,1])
            TP_unshown = np.sum(total_detections[pred_indexes_unshown,1])
            p = np.nan_to_num(TP/len(pred_indexes))
            r = np.nan_to_num(TP/(n_gt_labels_visible + n_gt_labels_unshown))
            p_visible = np.nan_to_num(TP_visible/len(pred_indexes_visible))
            r_visible = np.nan_to_num(TP_visible/n_gt_labels_visible)
            p_unshown = np.nan_to_num(TP_unshown/len(pred_indexes_unshown))
            r_unshown = np.nan_to_num(TP_unshown/n_gt_labels_unshown)
            precision[-1].append(p)
            recall[-1].append(r)
            precision_visible[-1].append(p_visible)
            recall_visible[-1].append(r_visible)
            precision_unshown[-1].append(p_unshown)
            recall_unshown[-1].append(r_unshown)

    precision = np.array(precision).transpose()
    recall = np.array(recall).transpose()
    precision_visible = np.array(precision_visible).transpose()
    recall_visible = np.array(recall_visible).transpose()
    precision_unshown = np.array(precision_unshown).transpose()
    recall_unshown = np.array(recall_unshown).transpose()



    # Sort the points based on the recall, class per class
    for i in np.arange(num_classes):
        index_sort = np.argsort(recall[:,i])
        precision[:,i] = precision[index_sort,i]
        recall[:,i] = recall[index_sort,i]

    # Sort the points based on the recall, class per class
    for i in np.arange(num_classes):
        index_sort = np.argsort(recall_visible[:,i])
        precision_visible[:,i] = precision_visible[index_sort,i]
        recall_visible[:,i] = recall_visible[index_sort,i]

    # Sort the points based on the recall, class per class
    for i in np.arange(num_classes):
        index_sort = np.argsort(recall_unshown[:,i])
        precision_unshown[:,i] = precision_unshown[index_sort,i]
        recall_unshown[:,i] = recall_unshown[index_sort,i]

    return precision, recall, precision_visible, recall_visible, precision_unshown, recall_unshown

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

    return np.mean(mAP_per_class), mAP_per_class

# Tight: (SNv3): np.arange(5)*1 + 1
# Loose: (SNv1/v2): np.arange(12)*5 + 5
def delta_curve(targets, closests, detections,  framerate, deltas=np.arange(5)*1 + 1):

    mAP = list()
    mAP_per_class = list()
    mAP_visible = list()
    mAP_per_class_visible = list()
    mAP_unshown = list()
    mAP_per_class_unshown = list()

    for delta in tqdm(deltas*framerate):

        precision, recall, precision_visible, recall_visible, precision_unshown, recall_unshown = compute_precision_recall_curve(targets, closests, detections, delta)


        tmp_mAP, tmp_mAP_per_class = compute_mAP(precision, recall)
        mAP.append(tmp_mAP)
        mAP_per_class.append(tmp_mAP_per_class)
        tmp_mAP_visible, tmp_mAP_per_class_visible = compute_mAP(precision_visible, recall_visible)
        mAP_visible.append(tmp_mAP_visible)
        mAP_per_class_visible.append(tmp_mAP_per_class_visible)
        tmp_mAP_unshown, tmp_mAP_per_class_unshown = compute_mAP(precision_unshown, recall_unshown)
        mAP_unshown.append(tmp_mAP_unshown)
        mAP_per_class_unshown.append(tmp_mAP_per_class_unshown)

    return mAP, mAP_per_class, mAP_visible, mAP_per_class_visible, mAP_unshown, mAP_per_class_unshown


def average_mAP(targets, detections, closests, framerate=2, deltas=np.arange(5)*1 + 1):


    mAP, mAP_per_class, mAP_visible, mAP_per_class_visible, mAP_unshown, mAP_per_class_unshown = delta_curve(targets, closests, detections, framerate, deltas)
    
    if len(mAP) == 1:
        return mAP[0], mAP_per_class[0], mAP_visible[0], mAP_per_class_visible[0], mAP_unshown[0], mAP_per_class_unshown[0]
    
    # Compute the average mAP
    integral = 0.0
    for i in np.arange(len(mAP)-1):
        integral += (mAP[i]+mAP[i+1])/2
    a_mAP = integral/((len(mAP)-1))

    integral_visible = 0.0
    for i in np.arange(len(mAP_visible)-1):
        integral_visible += (mAP_visible[i]+mAP_visible[i+1])/2
    a_mAP_visible = integral_visible/((len(mAP_visible)-1))

    integral_unshown = 0.0
    for i in np.arange(len(mAP_unshown)-1):
        integral_unshown += (mAP_unshown[i]+mAP_unshown[i+1])/2
    a_mAP_unshown = integral_unshown/((len(mAP_unshown)-1))
    a_mAP_unshown = a_mAP_unshown*17/13

    a_mAP_per_class = list()
    for c in np.arange(len(mAP_per_class[0])):
        integral_per_class = 0.0
        for i in np.arange(len(mAP_per_class)-1):
            integral_per_class += (mAP_per_class[i][c]+mAP_per_class[i+1][c])/2
        a_mAP_per_class.append(integral_per_class/((len(mAP_per_class)-1)))

    a_mAP_per_class_visible = list()
    for c in np.arange(len(mAP_per_class_visible[0])):
        integral_per_class_visible = 0.0
        for i in np.arange(len(mAP_per_class_visible)-1):
            integral_per_class_visible += (mAP_per_class_visible[i][c]+mAP_per_class_visible[i+1][c])/2
        a_mAP_per_class_visible.append(integral_per_class_visible/((len(mAP_per_class_visible)-1)))

    a_mAP_per_class_unshown = list()
    for c in np.arange(len(mAP_per_class_unshown[0])):
        integral_per_class_unshown = 0.0
        for i in np.arange(len(mAP_per_class_unshown)-1):
            integral_per_class_unshown += (mAP_per_class_unshown[i][c]+mAP_per_class_unshown[i+1][c])/2
        a_mAP_per_class_unshown.append(integral_per_class_unshown/((len(mAP_per_class_unshown)-1)))

    return a_mAP, a_mAP_per_class, a_mAP_visible, a_mAP_per_class_visible, a_mAP_unshown, a_mAP_per_class_unshown
