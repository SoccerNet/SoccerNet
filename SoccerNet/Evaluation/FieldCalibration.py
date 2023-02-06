import zipfile
import argparse
import numpy as np
import json

from SoccerNet.Evaluation.utils_calibration import mirror_labels, evaluate_detection_prediction, scale_points


def evaluate(gt_zip, prediction_zip, threshold=10, width=960, height=540, folder=""):
    gt_archive = zipfile.ZipFile(gt_zip, 'r')
    prediction_archive = zipfile.ZipFile(prediction_zip, 'r')
    gt_jsons = gt_archive.namelist()

    accuracies = []
    precisions = []
    recalls = []
    dict_errors = {}
    per_class_confusion_dict = {}
    total_frames = 0
    missed = 0
    for gt_json in gt_jsons:
        split, name = gt_json.split("/")

        if folder == "":
            pred_name = f"extremities_{name}"
        else:
            pred_name = f"{folder}/extremities_{name}"

        # pred_name = f"{split}/extremities_{name}"
        # print(pred_name)

        if pred_name not in prediction_archive.namelist():
            accuracies.append(0.)
            precisions.append(0.)
            recalls.append(0.)
            continue
        prediction = prediction_archive.read(pred_name)
        prediction = json.loads(prediction.decode("utf-8"))
        gt = gt_archive.read(gt_json)
        gt = json.loads(gt.decode('utf-8'))

        predictions = scale_points(prediction, width, height)
        line_annotations = scale_points(gt, width, height)

        img_prediction = predictions
        img_groundtruth = line_annotations
        confusion1, per_class_conf1, reproj_errors1 = evaluate_detection_prediction(img_prediction,
                                                                                    img_groundtruth,
                                                                                    threshold)
        confusion2, per_class_conf2, reproj_errors2 = evaluate_detection_prediction(img_prediction,
                                                                                    mirror_labels(
                                                                                        img_groundtruth),
                                                                                    threshold)

        accuracy1, accuracy2 = 0., 0.
        if confusion1.sum() > 0:
            accuracy1 = confusion1[0, 0] / confusion1.sum()

        if confusion2.sum() > 0:
            accuracy2 = confusion2[0, 0] / confusion2.sum()

        if accuracy1 > accuracy2:
            accuracy = accuracy1
            confusion = confusion1
            per_class_conf = per_class_conf1
            reproj_errors = reproj_errors1
        else:
            accuracy = accuracy2
            confusion = confusion2
            per_class_conf = per_class_conf2
            reproj_errors = reproj_errors2

        accuracies.append(accuracy)
        if confusion[0, :].sum() > 0:
            precision = confusion[0, 0] / (confusion[0, :].sum())
            precisions.append(precision)
        if (confusion[0, 0] + confusion[1, 0]) > 0:
            recall = confusion[0, 0] / (confusion[0, 0] + confusion[1, 0])
            recalls.append(recall)

        for line_class, errors in reproj_errors.items():
            if line_class in dict_errors.keys():
                dict_errors[line_class].extend(errors)
            else:
                dict_errors[line_class] = errors

        for line_class, confusion_mat in per_class_conf.items():
            if line_class in per_class_confusion_dict.keys():
                per_class_confusion_dict[line_class] += confusion_mat
            else:
                per_class_confusion_dict[line_class] = confusion_mat
    results = {}
    results["meanRecall"] = np.mean(recalls)
    results["meanPrecision"] = np.mean(precisions)
    results["meanAccuracies"] = np.mean(accuracies)

    for line_class, confusion_mat in per_class_confusion_dict.items():
        class_accuracy = confusion_mat[0, 0] / confusion_mat.sum()
        class_recall = confusion_mat[0, 0] / (confusion_mat[0, 0] + confusion_mat[1, 0])
        class_precision = confusion_mat[0, 0] / (confusion_mat[0, 0] + confusion_mat[0, 1])
        results[f"{line_class}Precision"] = class_precision
        results[f"{line_class}Recall"] = class_recall
        results[f"{line_class}Accuracy"] = class_accuracy
    return results

