import os
import json
import numpy as np
import argparse
import trackeval  # pip install git+https://github.com/JonathonLuiten/TrackEval.git

# import sys
import os
import argparse
from multiprocessing import freeze_support
import zipfile
import shutil
from pathlib import Path

def evaluate(groundtruth_filename, prediction_filename, split="test"):

    if os.path.exists("./temp/"):
        shutil.rmtree("./temp/")


    freeze_support()

    # Command line interface:
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False
    default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    default_metrics_config = {'METRICS': [
        'HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}
    config = {**default_eval_config, **default_dataset_config,
            **default_metrics_config}  # Merge default configs
    parser = argparse.ArgumentParser()
    for setting in config.keys():
        if type(config[setting]) == list or type(config[setting]) == type(None):
            parser.add_argument("--" + setting, nargs='+')
        else:
            parser.add_argument("--" + setting)

    # parser.add_argument('--TRACKERS_FOLDER_ZIP', type=str,  default="")
    # parser.add_argument('--GT_FOLDER_ZIP', type=str, default="")

    # parser.add_argument('--BENCHMARK', type=str, default='SNMOT' )
    # parser.add_argument('--DO_PREPROC', type=str, default='False' )
    # parser.add_argument('--SEQMAP_FILE', type=str, default='SNMOT-test.txt')
    # parser.add_argument('--TRACKERS_TO_EVAL', type=str, default='test' )
    # parser.add_argument('--SPLIT_TO_EVAL', type=str, default='test' )
    # parser.add_argument('--OUTPUT_SUB_FOLDER', type=str, default='eval_results' )
    # --TRACKERS_FOLDER_ZIP soccernet_mot_results.zip 
    # --GT_FOLDER_ZIP gt.zip

    # args = parser.parse_args().__dict__
    args = parser.parse_args()

    # args.TRACKER_FOLDER_ZIP = prediction_filename
    # args.GT_FOLDER_ZIP = groundtruth_filename

    # import pdb; pdb.set_trace()
    # if not empty ..., extract and modify trackers folder
    # assert len(args.TRACKER_FOLDER_ZIP) > 0
    # assert len(args.GT_FOLDER_ZIP) > 0

    args.BENCHMARK = "SNMOT"
    args.DO_PREPROC= "False" #[False]
    args.SEQMAP_FILE = [Path(__file__).parent / os.path.join("..","data",f"SNMOT-{split}.txt")]
    # print(args.SEQMAP_FILE)
    args.TRACKERS_TO_EVAL=["test"]
    args.SPLIT_TO_EVAL=f"{split}"
    args.OUTPUT_SUB_FOLDER="eval_results"

    # os.mkdir('./temp')f
    # os.mkdir('./temp/gt')
    # os.mkdir('./temp/SNMOT-test/')
    # os.mkdir('./temp/SNMOT-test/test')
    # os.mkdir('./temp/SNMOT-test/test/data')
    os.makedirs('./temp/SNMOT-test/test/data', exist_ok=True)

    with zipfile.ZipFile(prediction_filename, 'r') as zip_ref:
        zip_ref.extractall(f'./temp/SNMOT-{split}/test/data')
    with zipfile.ZipFile(groundtruth_filename, 'r') as zip_ref:
        zip_ref.extractall('./temp/gt/SNMOT-test_0')

    shutil.move(f'./temp/gt/SNMOT-test_0/{split}-evalAI/', f'./temp/gt/SNMOT-{split}/')

    args.TRACKERS_FOLDER = './temp'
    args.GT_FOLDER = './temp/gt'

    args = args.__dict__
    args['SEQMAP_FILE'] = args['SEQMAP_FILE'][0]
    # args['SEQMAP_FILE'] = f"SNMOT-{split}.txt" #args['SEQMAP_FILE'][0]
    args.pop('TRACKERS_FOLDER_ZIP', None)
    args.pop('GT_FOLDER_ZIP', None)

    for setting in args.keys():
        if args[setting] is not None:
            if type(config[setting]) == type(True):
                if args[setting] == 'True':
                    x = True
                elif args[setting] == 'False':
                    x = False
                else:
                    raise Exception('Command line parameter ' +
                                    setting + 'must be True or False')
            elif type(config[setting]) == type(1):
                x = int(args[setting])
            elif type(args[setting]) == type(None):
                x = None
            elif setting == 'SEQ_INFO':
                x = dict(zip(args[setting], [None]*len(args[setting])))
            else:
                x = args[setting]
            config[setting] = x
    eval_config = {k: v for k, v in config.items(
    ) if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items(
    ) if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items(
    ) if k in default_metrics_config.keys()}

    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)

    if os.path.exists("./temp/"):
        shutil.rmtree("./temp/")

    # print("output_msg", output_msg)
    # print("output_res", output_res.keys())
    # print("MotChallenge2DBox", output_res["MotChallenge2DBox"].keys())
    # print("test", output_res["MotChallenge2DBox"]["test"].keys())
    # print("COMBINED_SEQ",
    #       output_res["MotChallenge2DBox"]["test"]["COMBINED_SEQ"])
    # print("pedestrian",
    #       output_res["MotChallenge2DBox"]["test"]["COMBINED_SEQ"]["pedestrian"])
    # print("HOTA",
    #       output_res["MotChallenge2DBox"]["test"]["COMBINED_SEQ"]["pedestrian"]["HOTA"])

    # print("HOTA",
    #       output_res["MotChallenge2DBox"]["test"]["COMBINED_SEQ"]["pedestrian"]["HOTA"]["HOTA"])
    HOTA = output_res["MotChallenge2DBox"]["test"]["COMBINED_SEQ"]["pedestrian"]["HOTA"]["HOTA"]
    HOTA= sum(HOTA)/len(HOTA)
    # print("HOTA", HOTA)
    DetA = output_res["MotChallenge2DBox"]["test"]["COMBINED_SEQ"]["pedestrian"]["HOTA"]["DetA"]
    DetA = sum(DetA)/len(DetA)
    # print("DetA", DetA)
    AssA = output_res["MotChallenge2DBox"]["test"]["COMBINED_SEQ"]["pedestrian"]["HOTA"]["AssA"]
    AssA = sum(AssA)/len(AssA)
    # print("AssA", AssA)
    # print("HOTA", sum(AssA)/len(AssA)*sum(DetA)/len(DetA))

    
    
    performance_metrics = {}
    performance_metrics["HOTA"] = HOTA
    performance_metrics["DetA"] = DetA
    performance_metrics["AssA"] = AssA

    return performance_metrics
    
