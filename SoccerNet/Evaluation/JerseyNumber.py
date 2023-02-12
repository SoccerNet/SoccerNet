import json
import numpy as np
from sklearn.metrics import accuracy_score


def evaluate(ground_truth_file, predictions_file):
    pred_dict = json.load(open(predictions_file))
    gt_dict = json.load(open(ground_truth_file))
    pred_array = []
    gt_array = []
    num_players = len(gt_dict)
    for k in gt_dict.keys():
        gt_array.append(gt_dict[k])
        assert k in pred_dict, f"Jersey number prediction for player {k} is missing. Prediction file should contain a " \
                          f"dictionnary with one entry for each of the {num_players} players, with the player id " \
                          f"(string) as key and the player jersey number (int) as value (-1 if no jersey number " \
                          f"visible)."
        number = pred_dict[k]
        assert isinstance(number, int), f"Player jersey number should be provided as a int, but a {type(number)} was " \
                                        f"provided for player {k}."
        pred_array.append(number)
    pred_array = np.array(pred_array)
    gt_array = np.array(gt_array)
    return {"Accuracy": accuracy_score(gt_array, pred_array)}

