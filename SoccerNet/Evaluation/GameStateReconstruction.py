import trackeval  # pip install git+https://github.com/SoccerNet/sn-trackeval.git

import os
import argparse
import zipfile
import shutil
from multiprocessing import freeze_support


def evaluate(groundtruth_directory, prediction_filename, split="test"):

    if os.path.exists("./temp/"):
        shutil.rmtree("./temp/")

    freeze_support()

    tracker_name = "predictions"

    zip_path = prediction_filename
    target_dir = f'./temp/SoccerNetGS-{split}/{tracker_name}'

    # Make sure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get a list of all files in the ZIP archive
        for file_name in zip_ref.namelist():
            # Check if the file is a .json file
            if file_name.endswith('.json'):
                # Define the target path for the .json file
                # os.path.basename(file_name) gets the file name itself, ignoring directories
                target_path = os.path.join(target_dir, os.path.basename(file_name))

                # Extract the file to the specific path
                # However, since zipfile.extract() extracts with the full path, we'll read and then write
                # the file to achieve the desired structure
                with zip_ref.open(file_name) as source_file:
                    with open(target_path, 'wb') as target_file:
                        # Copy the file content to the target directory
                        target_file.write(source_file.read())

    # Extract GT zipped folder
    zip_path = groundtruth_directory
    target_dir = f'./temp/SoccerNetGS-{split}/groundtruth'

    # Make sure the target directory exists
    os.makedirs(os.path.join(target_dir, split), exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get a list of all files in the ZIP archive
        for file_name in zip_ref.namelist():
            # Check if the file is a .json file
            if file_name.endswith('.json'):
                # Define the target path for the .json file
                # os.path.basename(file_name) gets the file name itself, ignoring directories
                # print(file_name)
                target_path = os.path.join(target_dir, split, file_name)
                # print(os.path.dirname(target_path))
                os.makedirs(os.path.dirname(target_path), exist_ok=True)

                # Extract the file to the specific path
                # However, since zipfile.extract() extracts with the full path, we'll read and then write
                # the file to achieve the desired structure
                with zip_ref.open(file_name) as source_file:
                    with open(target_path, 'wb') as target_file:
                        # Copy the file content to the target directory
                        target_file.write(source_file.read())

    groundtruth_directory = target_dir

    # Extract GT zipped folder?

    # zip_path = groundtruth_directory
    # target_dir = f'./temp/SoccerNetGS-{split}/groundtruth'

    # # Make sure the target directory exists
    # os.makedirs(target_dir, exist_ok=True)

    # with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    #     # Get a list of all files in the ZIP archive
    #     for file_name in zip_ref.namelist():
    #         # Check if the file is a .json file
    #         if file_name.endswith('.json'):
    #             # Define the target path for the .json file
    #             # os.path.basename(file_name) gets the file name itself, ignoring directories
    #             print(file_name)
    #             target_path = os.path.join(target_dir, file_name)
    #             print(os.path.dirname(target_path))
    #             os.makedirs(os.path.dirname(target_path), exist_ok=True)

    #             # Extract the file to the specific path
    #             # However, since zipfile.extract() extracts with the full path, we'll read and then write
    #             # the file to achieve the desired structure
    #             with zip_ref.open(file_name) as source_file:
    #                 with open(target_path, 'wb') as target_file:
    #                     # Copy the file content to the target directory
    #                     target_file.write(source_file.read())


    # Forked from run_soccernet_gs.py :
    # Command line interface
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config['DISPLAY_LESS_PROGRESS'] = False
    dataset_config = trackeval.datasets.SoccerNetGS.get_default_dataset_config()
    metrics_config = {'METRICS': ['HOTA', 'Identity'], 'THRESHOLD': 0.5}


    # updating SoccerNet config
    dataset_config['GT_FOLDER'] = groundtruth_directory
    dataset_config['TRACKERS_FOLDER'] = './temp'
    dataset_config['TRACKER_SUB_FOLDER'] = ""
    dataset_config['TRACKERS_TO_EVAL'] = ["predictions"]
    dataset_config['SPLIT_TO_EVAL'] = split
    eval_config['TIME_PROGRESS'] = False
    eval_config['USE_PARALLEL'] = False
    eval_config['PRINT_RESULT'] = True
    eval_config['PRINT_ONLY_COMBINED'] = True
    eval_config['OUTPUT_SUMMARY'] = True
    eval_config['OUTPUT_DETAILED'] = False
    eval_config['PLOT_CURVES'] = False

    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.SoccerNetGS(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)

    if os.path.exists("./temp/"):
        shutil.rmtree("./temp/")

    task = "SoccerNetGS"
    HOTA = output_res[task][tracker_name]["SUMMARIES"]["cls_comb_det_av"]["HOTA"]["HOTA"]
    DetA = output_res[task][tracker_name]["SUMMARIES"]["cls_comb_det_av"]["HOTA"]["DetA"]
    AssA = output_res[task][tracker_name]["SUMMARIES"]["cls_comb_det_av"]["HOTA"]["AssA"]

    HOTA = float(HOTA)
    DetA = float(DetA)
    AssA = float(AssA)

    performance_metrics = {}
    performance_metrics["GS-HOTA"] = HOTA/100
    performance_metrics["GS-DetA"] = DetA/100
    performance_metrics["GS-AssA"] = AssA/100

    # display results
    print('** Game State Results **')
    print('GS-HOTA: {}%'.format(HOTA))
    print('GS-DetA: {}%'.format(DetA))
    print('GS-AssA: {}%'.format(AssA))

    return performance_metrics


# if __name__ == "__main__":
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-pr", "--prediction",
#                         help="prediction zip",
#                         type=str,
#                         default="")
#     parser.add_argument("-gt", "--groundtruth",
#                         help="groundtruth folder",
#                         type=str,
#                         default="")
#     parser.add_argument("-sp", "--split",
#                         help="set split",
#                         type=str,
#                         default="")
#     parsed_args = parser.parse_args()
#
#     evaluate(parsed_args.groundtruth, parsed_args.prediction, parsed_args.split)
