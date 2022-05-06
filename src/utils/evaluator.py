import os.path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import numpy as np
from dataclasses import dataclass
import dataclasses
import json
import pathlib
import pickle
from tqdm import tqdm

from src.data.orbit_few_shot_video_classification import VideoInfo
from src.data.utils import _ndarray_to_list


@dataclass
class ClipResult:
    video_name: str
    user_id: str
    gt_label: int
    gt_object_name: str
    per_clip_feature: np.ndarray
    per_frame_filenames: List[str]


@dataclass
class VideoResult:
    video_name: str
    user_id: str
    gt_label: int
    gt_object_name: str
    per_frame_features: List[np.ndarray]
    per_frame_predictions: List[int]
    per_frame_scores: List[float]
    per_frame_filenames: List[str]
    video_frame_accuracy: float = 0.0


@dataclass
class EpisodeResult:
    query_video_results: List[VideoResult] = None
    support_clips_results: List[ClipResult] = None
    object_names: List[str] = None
    prototypes: List[np.ndarray] = None


@dataclass
class FEATEpisodeResult(EpisodeResult):
    original_prototypes: List[np.ndarray] = None


def calculate_confidence_interval(scores):
    return (1.96 * np.std(scores)) / np.sqrt(len(scores))


def calculate_frame_accuracy(video_gt_label: int,
                             per_frame_predictions: List[int]):
    per_frame_predictions = np.array(per_frame_predictions)
    per_frame_correct = np.equal(video_gt_label, per_frame_predictions).astype(int)
    average_frame_accuracy = np.mean(per_frame_correct)
    return average_frame_accuracy


def calculate_video_accuracy(video_gt_label: int,
                             per_frame_predictions: List[int]):
    per_frame_predictions = np.array(per_frame_predictions)
    most_freq_prediction = np.bincount(per_frame_predictions).argmax()
    return 1.0 if most_freq_prediction == video_gt_label else 0.0


def calculate_frames_to_recognition(video_gt_label: int,
                                    per_frame_predictions: List[int]):
    per_frame_predictions = np.array(per_frame_predictions)
    correct = np.where(video_gt_label == per_frame_predictions)[0]
    if len(correct) > 0:
        return correct[0] / len(per_frame_predictions)  # first correct frame index / num_frames
    else:
        return 1.0  # no correct predictions; last frame index / num_frames (i.e. 1.0)


_metrics_function = {"per_frame_accuracy": calculate_frame_accuracy,
                     "frames_to_recognition": calculate_frames_to_recognition,
                     "video_accuracy": calculate_video_accuracy}


class EpisodeEvaluator:
    def __init__(self, save_dir=None):
        if not save_dir:
            raise ValueError("Lack of save_dir to save feature maps and visualized figures")
        pathlib.Path(save_dir).mkdir(exist_ok=True, parents=True)
        self.save_dir = save_dir
        self.episode = None
        self.reset()

    def reset(self, ):
        self.episode = EpisodeResult()
        self.episode.query_video_results = []
        self.episode.support_clips_results = []

    def register_object_category(self, object_categories: List[str]):
        self.episode.object_names = object_categories

    def register_prototypes(self, prototypes: np.ndarray):
        self.episode.prototypes = [prototype for prototype in prototypes]

    def add_multiple_clips_results(self, clips_features: np.ndarray, clips_filenames: List[List[str]],
                               clips_labels: np.ndarray):
        for clip_feature, clip_filenames, clip_label in zip(clips_features, clips_filenames, clips_labels):
            clip_info = VideoInfo(clip_filenames[0])
            clip_result = ClipResult(video_name=clip_info.video_name,
                                     user_id=clip_info.user_id,
                                     gt_label=int(clip_label),
                                     gt_object_name=clip_info.object_category,
                                     per_clip_feature=clip_feature,
                                     per_frame_filenames=clip_filenames)
            self.episode.support_clips_results.append(clip_result)

    def add_video_result(self,
                         per_frame_prediction_scores: np.ndarray,
                         per_frame_features: np.ndarray,
                         frame_filenames: List[str],
                         video_gt_label: int,
                         video_frame_accuracy: float):
        # Get metadata
        video_info = VideoInfo(frame_filenames[0])

        # remove duplicate frames at the last
        frame_paths, unique_idxs = np.unique(frame_filenames, return_index=True)
        frame_filenames = [frame_filenames[idx] for idx in unique_idxs]
        per_frame_prediction_scores = per_frame_prediction_scores[unique_idxs]
        per_frame_features = per_frame_features[unique_idxs]

        # Prepare per video result
        per_frame_predictions = _ndarray_to_list(per_frame_prediction_scores.argmax(axis=-1))
        per_frame_confidence_scores = _ndarray_to_list(per_frame_prediction_scores.max(axis=-1))
        per_frame_features = _ndarray_to_list(per_frame_features)

        assert len(per_frame_predictions) == len(per_frame_confidence_scores) == len(
            per_frame_features), "The number of per frame predictions must be equal to that of per frame features "

        video_result = VideoResult(video_name=video_info.video_name,
                                   user_id=video_info.user_id,
                                   gt_label=video_gt_label,
                                   gt_object_name=video_info.object_category,
                                   per_frame_predictions=per_frame_predictions,
                                   per_frame_scores=per_frame_confidence_scores,
                                   per_frame_features=per_frame_features,
                                   per_frame_filenames=frame_filenames,
                                   video_frame_accuracy=video_frame_accuracy)
        self.episode.query_video_results.append(video_result)

    def compute_statistics(self):
        total_frame_accuracy = [video_result.video_frame_accuracy for video_result in self.episode.query_video_results]
        total_metrics = np.array(total_frame_accuracy)
        mean_metric = np.mean(total_metrics)
        confidence_interval = calculate_confidence_interval(total_metrics)
        print("User_id = {}, The average frame accuracy across all {} testing videos  = {}, confidence_interval = {}".
              format(self.episode.query_video_results[0].user_id, len(total_metrics), mean_metric, confidence_interval))

    def save_to_disk(self):
        if not self.episode.query_video_results:
            raise ValueError("Error! No any video result has been registered")
        user_id = self.episode.query_video_results[0].user_id
        filename = os.path.join(self.save_dir, user_id + ".p")
        with open(filename, "wb") as f:
            pickle.dump(self.episode, f)


class FEATEpisodeEvaluator(EpisodeEvaluator):
    def reset(self, ):
        self.episode = FEATEpisodeResult()
        self.episode.query_video_results = []
        self.episode.support_clips_results = []

    def register_original_prototypes(self, original_prototypes: np.ndarray):
        self.episode.original_prototypes = [prototype for prototype in original_prototypes]


def convert_results_in_submission_format(exp_path: str):
    """
    Example: https://eval.ai/web/challenges/challenge-page/1438/submission
        {
            "P177": {
                "user_objects": ["bag", "hairbrush", "hat", "keys", "phone", "secateurs", "tv remote"],
                "user_videos" : {
                    "P177--bag--clutter--Zj_1HvmNWejSbmYf_m4YzxHhSUUl-ckBtQ-GSThX_4E": [0, 0, 0, 0, 2, …., 0, 0, 0, 6, 0 ],
                    "P177--hairbrush--clutter--GL4DgGhfREqG9d3j3RcvY3xZuFMaz3QcKfSU0gLUwJI": [1, 1, 2, 2, 2, … ,6, 6, 1, 1, 1],
                     #for all clutter videos for test user P177
                }
            },

            "P999": {
                … #objects and predictions for all clutter videos for final test user
            }
        }
    """
    submission_results = {}

    episode_filenames = list(pathlib.Path(exp_path).rglob("*.p"))
    for filename in tqdm(episode_filenames):
        with open(filename, "rb") as f:
            episode_result = pickle.load(f)
        user_id = episode_result.query_video_results[0].user_id
        submission_results[user_id] = {}
        submission_results[user_id]["user_objects"] = episode_result.object_names
        submission_results[user_id]["user_videos"] = {}
        for video_result in episode_result.query_video_results:
            predictions = [int(e)for e in video_result.per_frame_predictions]
            submission_results[user_id]["user_videos"][
                video_result.video_name] = predictions

    target_filename = pathlib.Path(exp_path, "orbit_submission.json")
    with target_filename.open(mode='w') as f:
        json.dump(submission_results, f)
    print("ORBIT Challenge 20220 Format, Converted Success！{}".format(target_filename))


def compute_average_frame_accuracy_across_videos(exp_path: str):
    total_frame_accuracy = []
    import torchmetrics
    tmp = []
    import torch
    episode_filenames = list(pathlib.Path(exp_path).rglob("*.p"))
    if not episode_filenames:
        raise  ValueError("No result of any one testing user  ")
    for filename in tqdm(episode_filenames):
        with open(filename, "rb") as f:
            episode_result = pickle.load(f)
            # Calculate frame accuray of each video
            total_frame_accuracy.extend(
                [calculate_frame_accuracy(video_result.gt_label, video_result.per_frame_predictions)
                 for video_result in episode_result.query_video_results])
            tmp.extend([torchmetrics.functional.accuracy(torch.tensor(video_result.per_frame_predictions),
                                                         torch.tensor(
                                                             np.array([video_result.gt_label] * len(video_result.per_frame_predictions))
                                                                    )
                                                        )
                 for video_result in episode_result.query_video_results])

    total_metrics = np.array(total_frame_accuracy)
    mean_metric = np.mean(total_metrics)
    confidence_interval = calculate_confidence_interval(total_metrics)
    print("The average frame accuracy across all {} testing videos  = {}, confidence_interval = {}".
          format(len(total_metrics), mean_metric, confidence_interval))
