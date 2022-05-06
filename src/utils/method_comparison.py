import os.path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import numpy as np
import pandas as pd
import pathlib
import pickle
from tqdm import tqdm
from src.utils.evaluator import calculate_frame_accuracy


def prepare_user_centric_testing_results(exp_path: str, method="protonet"):
    """
    user_id, frame_accuracy, method
    """
    tmp = {"user_id": [],
           "num_videos": [],
           "frame_accuracy": [],
           "method": []}
    total_frame_accuracy = []
    episode_filenames = list(pathlib.Path(exp_path).rglob("*.p"))
    if not episode_filenames:
        raise  ValueError("No result of any one testing user  ")
    for filename in tqdm(episode_filenames):
        with open(filename, "rb") as f:
            episode_result = pickle.load(f)
            # Calculate frame accuracy of each video
        total_frame_accuracy = [calculate_frame_accuracy(video_result.gt_label, video_result.per_frame_predictions)
                 for video_result in episode_result.query_video_results]
        tmp["user_id"].append(episode_result.query_video_results[0].user_id)
        tmp["num_videos"].append(len(episode_result.query_video_results))
        tmp["frame_accuracy"].append(np.array(total_frame_accuracy).mean())
        tmp["method"] = method
    # print("The average Frame Accuracy of using {} = {}".format(method, np.array(tmp["frame_accuracy"]).mean()))
    database_method = pd.DataFrame.from_dict(tmp)
    return database_method


def prepare_video_centric_testing_results(exp_path: str, method="protonet"):
    """
    user_id, frame_accuracy, method
    """
    tmp = {"video_name": [],
           "frame_accuracy": [],
           "method": [],
           "user_id": [],
           "video_length": []}
    episode_filenames = list(pathlib.Path(exp_path).rglob("*.p"))
    if not episode_filenames:
        raise  ValueError("No result of any one testing user  ")
    for filename in tqdm(episode_filenames):
        with open(filename, "rb") as f:
            episode_result = pickle.load(f)
            # Calculate frame accuracy of each video
            for video_result in episode_result.query_video_results:
                tmp["video_name"].append(video_result.video_name)
                tmp["frame_accuracy"].append(calculate_frame_accuracy(video_result.gt_label, video_result.per_frame_predictions))
                tmp["method"].append(method)
                tmp["user_id"].append(video_result.user_id)
                tmp["video_length"].append(len(video_result.per_frame_predictions))
    # print("The average Frame Accuracy of using {} = {}".format(method, np.array(tmp["frame_accuracy"]).mean()))
    database_method = pd.DataFrame.from_dict(tmp)
    return database_method


def collect_testing_results_from_multiple_methods(exp_paths: List[str], methods=List[str], mode="video"):
    if len(exp_paths) != len(methods):
        raise ValueError("the number of testing results should be same with the number of methods !")
    if mode == "user":
        func = prepare_user_centric_testing_results
    elif mode == "video":
        func = prepare_video_centric_testing_results
    else:
        raise NotImplementedError
    testing_result_databases = [func(exp_path, method)
                                for exp_path, method in zip(exp_paths, methods)]
    database = pd.concat(testing_result_databases)
    return database

if __name__ == '__main__':
    exp_paths = ["/home/ligu/projects/orbit_challenge_2022_refactor/logs/tb_logs/tt/testing_per_video_results",
                 "/home/ligu/projects/orbit_challenge_2022_refactor/logs/tb_logs/protonet_with_lite_official_orbit_baseline/testing_per_video_results"]
    methods = ["method_1",
               "method_2"]
    a = collect_testing_results_from_multiple_methods( exp_paths, methods, mode="video")
    b = 1

