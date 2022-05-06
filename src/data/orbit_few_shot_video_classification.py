import copy
import logging
import os
import json
import random
from pathlib import Path
import re
import numpy as np
import pandas as pd
from typing import Optional, Callable, Any, Dict, List, Tuple
from dataclasses import dataclass

from torch import Tensor
from tqdm import tqdm
from einops import rearrange

import torch
from torch.utils.data import DataLoader
import torchvision

from src.transforms import ApplyTransformToKey, MutuallyExclusiveLabel, Div255
from src.data.video import FrameVideo
from src.data.sampler import Sampler, make_sampler
from src.data.clip_sampler import make_clip_sampler, ClipSampler, MaxMultiClipSampler
from src.data.episode import Episode, UserCentricVideoInstance
from src.data.utils import _flatten_nested_list

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

VIDEO_TYPES = ["clean", "clutter"]

_FPS = 30
_MAX_NUM_CLIPS = 100000
_MAX_NUM_FRAMES = 100000


class VideoInfo:
    def __init__(self, filename_fullpath: str):
        self.filename_fullpath = filename_fullpath

    @property
    def video_name(self):
        return self.filename_fullpath.split("/")[-2]

    @property
    def user_id(self):
        return self.filename_fullpath.split("/")[-1].split("--")[0]

    @property
    def object_category(self):
        return self.filename_fullpath.split("/")[-1].split("--")[1]


def get_object_category_names_according_labels(videos_filenames, labels):
    if len(videos_filenames) != len(labels):
        raise ValueError("num of video sequences!= num of objects")
    mapping = {}
    for video_frame_filenames, label in zip(videos_filenames, labels):
        # object_name = filename[0].split("/")[-1].split("--")[1]
        object_name = video_frame_filenames[0].split("/")[6]
        if int(label) not in mapping:
            mapping[int(label)] = object_name
    mapping = {k: v for k, v in sorted(mapping.items(), key=lambda item: item[0])}

    def clean_object_category_names(object_list):
        """
            Satisfy requirements of submission
        """
        new_list = []
        for name in object_list:
            name = name.replace("-", " ")
            new_list.append(name)
        return new_list

    return clean_object_category_names([v for k, v in mapping.items()])


def prepare_ORBIT_dataset(root_data_path: str,
                          mode: str = "train"):
    """
        Arguments:
            root_data_path (str): the root folder of orbit dataset
            mode (str): train, val, test
        Return:
            video_database (pandas.DataFrame) Each row represents a video instance including 6 attributes:
                                "video_name, user, category_name, category_label, category_cluster_label, video_type, filename_fullpath"
            user_name
                * object_name
                    * video_type (clean or clutter)
                        * video_name
                            * user_name, category_name, category_label, category_cluster_label, video_type, filename_fullpath"
    """
    if not os.path.isdir(os.path.join(root_data_path, mode)):
        raise RuntimeError(
            "Cannot find the root_data_path of ORBIT dataset: {}".format(os.path.join(root_data_path, mode)))

    user_names = [user_name for user_name in os.listdir(os.path.join(root_data_path, mode))]
    logger.info("Total number of users = {} in mode {}".format(len(user_names), mode))
    with open(
            os.path.join(root_data_path, 'annotation', 'orbit_{:}_object_cluster_labels.json'.format(mode))) as in_file:
        video_name_to_cluster_label = json.load(in_file)

    # extract valid category_names from folders
    category_name_to_label = {}
    data_path = os.path.join(root_data_path, mode)
    label = 0
    for user_name in user_names:
        object_names = sorted(os.listdir(os.path.join(data_path, user_name)))
        for object_name in object_names:
            if object_name not in category_name_to_label:
                category_name_to_label[object_name] = label
                label += 1
    logger.info("Total number of objects = {} in mode {}".format(len(list(category_name_to_label.keys())), mode))

    # build a video database
    video_database = {}
    data_path = os.path.join(root_data_path, mode)

    cnt_user = 0
    cnt_video = 0
    for user_name in user_names:
        video_database[user_name] = {}
        object_names = sorted(os.listdir(os.path.join(data_path, user_name)))
        cnt_user += 1
        for object_name in object_names:
            video_database[user_name][object_name] = {}
            video_types = os.listdir(os.path.join(data_path, user_name, object_name))
            for video_type in video_types:
                if video_type not in VIDEO_TYPES:
                    raise ValueError("{} from collector {} has invalid video type: {}"
                                     .format(object_name, user_name, video_type))
                video_database[user_name][object_name][video_type] = []
                video_names = sorted(os.listdir(os.path.join(data_path, user_name, object_name, video_type)))
                for video_name in video_names:
                    video_frame_folder_fullpath = os.path.join(data_path, user_name, object_name, video_type,
                                                               video_name)
                    num_frames = len(list(Path(video_frame_folder_fullpath).rglob("*.jpg")))
                    if num_frames == 0:
                        logger.warning("There is no any single frame (*.jpg) in video {}".format(video_name))
                        continue
                    video_metadata = {"user_name": user_name,
                                      "video_name": video_name,
                                      "category_name": object_name,
                                      "category_label": category_name_to_label[object_name],
                                      "category_cluster_label": video_name_to_cluster_label[video_name],
                                      "video_type": video_type,
                                      "filename_fullpath": video_frame_folder_fullpath,
                                      "annotation_fullpath":
                                          os.path.join(root_data_path, "annotation", "orbit_extra_annotations",
                                                       mode, video_name + ".json")
                                      }
                    video_database[user_name][object_name][video_type].append(video_metadata)
                    cnt_video += 1
    logger.info("Total number of video instances = {} in mode {}".format(cnt_video, mode))
    return video_database


class ORBITDatasetVideoInstanceSampler:
    """
        How to sample support instances and query instance from each object in ORBIT User-Centric Dataset?

        Requirements:
            1. Make sure there is at least one sampled query instance from each object
            2. Make sure there are five sampled support instance from each object at minimum except
               condition 1 cannot be satisfied

    """
    MIN_NUM_SUPPORT_INSTANCE = 5
    MIN_NUM_QUERY_INSTANCE = 1

    def __init__(self,
                 support_instance_sampler: Sampler,
                 query_instance_sampler: Sampler,
                 query_video_type: str = "clean",
                 ) -> None:
        self.support_instance_sampler = support_instance_sampler
        self.query_instance_sampler = query_instance_sampler
        self.query_video_type = query_video_type

    def __call__(self, category_name_to_instances: Dict[str, List]) -> Tuple[List, List]:
        if "clean" not in category_name_to_instances.keys():
            raise ValueError("There is no clean video instance")
        if "clutter" not in category_name_to_instances.keys():
            raise ValueError("There is no clutter video instance")

        clean_video_instances = category_name_to_instances["clean"]
        clutter_video_instances = category_name_to_instances["clutter"]

        if self.query_video_type == "clean":
            if len(clean_video_instances) < self.MIN_NUM_SUPPORT_INSTANCE + self.MIN_NUM_QUERY_INSTANCE:
                logger.warning("The total number of clean videos  = {}, less than the minimum requirement: min num of "
                               "support = {}, min num of query = {}".
                               format(len(clean_video_instances), self.MIN_NUM_SUPPORT_INSTANCE,
                                      self.MIN_NUM_QUERY_INSTANCE))
            return self._sample_instance_from_only_clean_video_type(clean_video_instances)
        if self.query_video_type == "clutter":
            return self._sample_instance_from_individual_video_types(clean_video_instances, clutter_video_instances)

    def _sample_instance_from_individual_video_types(self,
                                                     clean_video_instances: List[Dict],
                                                     clutter_video_instances: List[Dict]):
        support_video_instances = self.support_instance_sampler(clean_video_instances)
        query_video_instances = self.query_instance_sampler(clutter_video_instances)
        return support_video_instances, query_video_instances

    def _sample_instance_from_only_clean_video_type(self, clean_video_instances: List[Dict]):
        total_num_clean_instance = len(clean_video_instances)
        num_support_video_instances = min(self.MIN_NUM_SUPPORT_INSTANCE,
                                          total_num_clean_instance - self.MIN_NUM_QUERY_INSTANCE)
        random.shuffle(clean_video_instances)

        support_video_instances = self.support_instance_sampler(clean_video_instances[:num_support_video_instances])
        query_video_instances = self.query_instance_sampler(clean_video_instances[num_support_video_instances:])
        return support_video_instances, query_video_instances


class UserCentricFewShotVideoClassificationDataset(torch.utils.data.Dataset):
    """
        User-centric Episode: all object classes from one sampled user
    """

    def __init__(
            self,
            video_database: Dict[str, Dict],
            category_sampler: Optional[Sampler],
            video_instance_sampler: ORBITDatasetVideoInstanceSampler,
            support_video_clip_sampler: ClipSampler,
            query_video_clip_sampler: ClipSampler,
            num_episodes_per_user: int = 50,
            user_cluster_label: bool = False,
            transform: Optional[Callable[[dict], Any]] = None,
            mode: str = "train",
            num_threads: int = 8,

    ) -> None:
        """
        Args:
            video_database (Dict[str, Dict]):
                user_name
                    * object_name
                        * video_type (clean or clutter)
                            * video_name
                                * user_name, category_name, category_label, category_cluster_label, video_type, filename_fullpath"
            category_sampler (Optional[Sampler]):
            video_instance_sampler (Optional[Sampler]):
            video_clip_sampler
            totol_num_episodes: the number of episodes will be generated in each epoch
            transform (Callable): transform_moduel module in "torchvison.transform_moduel"

        """

        self.video_database = video_database
        self.num_episode_per_user = num_episodes_per_user
        self.mode = mode

        self.category_sampler = category_sampler
        self.video_instance_sampler = video_instance_sampler
        self.support_video_clip_sampler = support_video_clip_sampler
        self.query_video_clip_sampler = query_video_clip_sampler
        self.label_key = "category_cluster_label" if user_cluster_label else "category_label"

        self.transform = transform
        self.num_threads = num_threads

        user_names = list(video_database.keys())
        user_names.sort()
        num_users = len(user_names)
        logger.info(
            '{} users in total, and each user has {} episodes.'.format(num_users, self.num_episode_per_user))

        logger.info("Start Sequence mode")
        self.episodes = []
        for user_name in tqdm(user_names):
            category_names = list(video_database[user_name].keys())
            category_name_to_instances = video_database[user_name]
            for _ in range(self.num_episode_per_user):
                self.episodes.append(self._generate_one_episode(category_names,
                                                                category_name_to_instances
                                                                ))

    def __getitem__(self, item) -> Dict:
        """
            TRAIN MODE:
            TEST MODE (TESTSET,VALSET):
        """
        episode = self.episodes[item]
        # STEP 1: Sample multiple clips from each video
        support_frames, support_labels, support_frame_filenames, support_annotations = \
            self._prepare_clips_tensor(
                video_filenames=[instance.filename for instance in episode.support_set],
                labels=[instance.label for instance in episode.support_set],
                annotation_filenames=[instance.annotation_filename for instance in episode.support_set],
                subset="support")

        query_frames, query_labels, query_frame_filenames, query_annotations = self._prepare_clips_tensor(
            video_filenames=[instance.filename for instance in episode.query_set],
            labels=[instance.label for instance in episode.query_set],
            annotation_filenames=[instance.annotation_filename for instance in episode.query_set],
            subset="query")

        category_names = list(set([instance.category_name for instance in episode.support_set]))
        user_names = [instance.user_name for instance in episode.support_set][0]

        # STEP 2: Shuffle sampled clips' indices
        support_frames, support_labels, support_frame_filenames, support_annotations = \
            self.__shuffle_clips_indices(support_frames, support_labels, support_frame_filenames, support_annotations)
        query_frames, query_labels, query_frame_filenames, query_annotations = \
            self.__shuffle_clips_indices(query_frames, query_labels, query_frame_filenames, query_annotations)

        raw_support_frames = copy.deepcopy(support_frames)
        raw_query_frames = copy.deepcopy(query_frames)

        # STEP 3: Apply Transform
        # TODO: Add video level augmentation techniques
        if self.transform:
            support_frames = rearrange(support_frames, "n t c h w  ->  (n t) c h w")
            query_frames = rearrange(query_frames, "n t c h w  ->  (n t) c h w")

            episode = {"support_image": support_frames,
                       "query_image": query_frames,
                       "episode_label": (support_labels, query_labels)}
            episode = self.transform(episode)
            support_frames, query_frames, (support_labels, query_labels) = \
                episode["support_image"], episode["query_image"], episode["episode_label"]

            support_frames = rearrange(
                support_frames, "(n t) c h w -> n t c h w ", t=self.support_video_clip_sampler.clip_length)
            query_frames = rearrange(
                query_frames, "(n t) c h w -> n t c h w ", t=self.query_video_clip_sampler.clip_length)

        # STEP 4: In validation set or test set, each query set consists of multiple video sequences rather than clips;
        #         Thus, for each video sequence, we need to covert its sampled multiple clips tensors into a video tensor
        if self.mode != "train":
            query_frames, query_labels, query_frame_filenames = \
                self.__convert_multi_clips_into_video_sequences(query_frames, query_labels, query_frame_filenames)

        episode_dict = {"support_frames": support_frames,  # Tensor[num_clip, clip_length, 3, 224, 224]
                        "raw_support_frames": raw_support_frames,
                        "support_targets": support_labels,  # Tensor[num_clip,]
                        "support_frame_filenames": support_frame_filenames,  # List[str]
                        # "raw_query_frames": raw_query_frames,
                        # Train: Tensor[num_clip, clip_length, 3, 224, 224]
                        # Test: List[Tensor(total_num_frames_one_video, 3, 224, 224)]
                        "query_frames": query_frames,
                        "query_targets": query_labels,
                        "query_frame_filenames": query_frame_filenames,
                        "category_names": category_names,  # List[str]
                        "user_id": user_names  # str
                        }

        return episode_dict

    def __len__(self):
        return len(self.episodes)

    def _generate_one_episode(self,
                              category_names: List[str],
                              category_name_to_instances: Dict[str, Dict],

                              ):
        sampled_category_names = self.category_sampler(category_names)

        support_set = []
        query_set = []
        for category_name in sampled_category_names:
            candidates = category_name_to_instances[category_name]
            support_video_instances, query_video_instances = self.video_instance_sampler(candidates)

            def prepare_instances_per_category(video_instances):
                return [UserCentricVideoInstance(
                    video_name=video_instance["video_name"],
                    user_name=video_instance["user_name"],
                    filename=video_instance["filename_fullpath"],
                    annotation_filename=video_instance["annotation_fullpath"],
                    label=video_instance[self.label_key],
                    category_name=video_instance["category_name"])
                    for video_instance in video_instances]

            support_set.extend(prepare_instances_per_category(support_video_instances))
            query_set.extend(prepare_instances_per_category(query_video_instances))

        support_set.sort(key=lambda x: x.label)
        query_set.sort(key=lambda x: x.label)
        return Episode(support_set=support_set, query_set=query_set)

    def _prepare_clips_tensor(self,
                              video_filenames: List[str],
                              labels: List[int],
                              annotation_filenames: List[str],
                              subset: str) \
            -> Tuple[Tensor, Tensor, object, List[Dict]]:
        multi_clips_frame = []
        multi_clips_frame_filenames = []
        multi_clips_labels = []
        multi_clips_annotations = []
        clip_sampler = self.support_video_clip_sampler if subset == "support" else self.query_video_clip_sampler
        for filename, label, annotation_filename in zip(video_filenames, labels, annotation_filenames):
            # 1. Prepare clips: Sample clips from one video sequence
            video = FrameVideo(video_folder_path=filename,
                               num_threads=self.num_threads,
                               clip_length=clip_sampler.clip_length)
            clip_info_list = clip_sampler(total_num_frame_video=video.total_num_frames)
            frames_tensor, frame_filenames \
                = video.get_multiple_clips(
                clips_frame_indices_list=[clip_info.clip_frame_indices for clip_info in clip_info_list])
            num_clips = frames_tensor.size(0)
            multi_clips_frame.append(frames_tensor)
            multi_clips_frame_filenames.append(frame_filenames)

            # 2. Prepare labels
            multi_clips_labels.append([label] * num_clips)

            # 3. Prepare annotations
            with open(annotation_filename) as f:
                per_frame_annotations = json.load(f)
            sampled_frame_annotations = [per_frame_annotations[Path(filename).name] for filename in
                                         _flatten_nested_list(frame_filenames)]
            clips_annotations = rearrange(np.array(sampled_frame_annotations), "(n t) -> n t",
                                          t=clip_sampler.clip_length).tolist()
            multi_clips_annotations.append(clips_annotations)

        multi_clips_frame_tensors = rearrange(torch.cat(multi_clips_frame, dim=0),
                                              "n t h w c ->  n t c h w")
        multi_clips_labels = torch.as_tensor(_flatten_nested_list(multi_clips_labels))
        multi_clips_frame_filenames = np.concatenate(multi_clips_frame_filenames, axis=0).tolist()
        multi_clips_annotations = _flatten_nested_list(multi_clips_annotations)
        return multi_clips_frame_tensors, multi_clips_labels, multi_clips_frame_filenames, multi_clips_annotations


    @staticmethod
    def __shuffle_clips_indices(frames: torch.FloatTensor, labels: torch.LongTensor, frame_filenames: List,
                                frame_annotations: List):
        indices = np.random.permutation(len(labels))
        frames = frames[indices]
        labels = labels[indices]
        frame_filenames = [frame_filenames[idx] for idx in indices]
        frame_annotations = [frame_annotations[idx] for idx in indices]
        return frames, labels, frame_filenames, frame_annotations

    @staticmethod
    def __convert_multi_clips_into_video_sequences(frames: torch.FloatTensor,
                                                   labels: torch.LongTensor,
                                                   frame_filenames: List[List[str]]):
        """
            frames: (torch.FloatTensor), shape = [num_clips, clip_length, c, h, w]
            labels: (torch.LongTensor), shape = [num_clips, ]
            frame_filenames: List[List[str]], shape = [num_clips[clip_length[str]]]
        """
        video_sequences_indices = {}
        for idx, (clip_frames, clip_label, clip_filenames) in enumerate(zip(frames, labels, frame_filenames)):
            video_name = VideoInfo(clip_filenames[0]).video_name
            if video_name not in video_sequences_indices:
                video_sequences_indices[video_name] = []
            video_sequences_indices[video_name].append(idx)

        video_frame_filenames = []
        video_frames = []
        video_labels = []
        for video_name in video_sequences_indices:
            indices = video_sequences_indices[video_name]
            indices.sort(key=lambda x: re.findall(r'\d+', Path(frame_filenames[x][0]).name)[-1])
            video_frames.append(torch.cat([frames[i] for i in indices], dim=0))
            video_labels.append(labels[indices[0]])
            video_frame_filenames.append(_flatten_nested_list([frame_filenames[i] for i in indices]))
        return video_frames, video_labels, video_frame_filenames


def ORBITUserCentricVideoFewShotClassification(
        data_path: str,
        mode: str = "train",
        object_sampler: str = "random",
        max_num_objects_per_user: int = 15,
        support_instance_sampler: str = "random",
        query_instance_sampler: str = "random",
        support_num_shot: Optional[int] = 5,
        query_num_shot: Optional[int] = 2,
        max_num_instance_per_object: int = 10,
        query_video_type: str = "clutter",
        support_clip_sampler: str = "random",
        query_clip_sampler: str = "max",
        video_subsample_factor: int = 1,
        video_clip_length: int = 8,
        max_num_clips_per_video: int = 10,
        max_num_sampled_frames_per_video: int = 1000,
        use_object_cluster_labels: bool = False,
        num_episodes_per_user: int = 50,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        num_threads: int = 8,
) -> torch.utils.data.Dataset:
    if mode not in ["train", "validation", "test"]:
        raise ValueError()

    if not os.path.exists(os.path.join(data_path, mode)):
        raise FileNotFoundError(os.path.join(data_path, mode))
    video_database = prepare_ORBIT_dataset(root_data_path=data_path, mode=mode)

    if object_sampler not in ["random", "max"]:
        raise ValueError()
    object_sampler = make_sampler(sampling_type=object_sampler,
                                  min_num_samples=2,
                                  max_num_samples=max_num_objects_per_user)
    support_instance_sampler = make_sampler(sampling_type=support_instance_sampler,
                                            num_samples=support_num_shot,
                                            max_num_samples=max_num_instance_per_object)
    query_instance_sampler = make_sampler(sampling_type=query_instance_sampler,
                                          num_samples=query_num_shot,
                                          max_num_samples=max_num_instance_per_object)
    video_instance_sampler = ORBITDatasetVideoInstanceSampler(support_instance_sampler=support_instance_sampler,
                                                              query_instance_sampler=query_instance_sampler,
                                                              query_video_type=query_video_type)
    support_clip_sampler = make_clip_sampler(sampling_type=support_clip_sampler,
                                             clip_length=video_clip_length,
                                             max_num_frame=max_num_sampled_frames_per_video,
                                             num_sampled_clips=max_num_clips_per_video,
                                             subsample_factor=video_subsample_factor)

    if mode == "train":
        query_clip_sampler = make_clip_sampler(sampling_type=query_clip_sampler,
                                               clip_length=video_clip_length,
                                               max_num_frame=max_num_sampled_frames_per_video,
                                               num_sampled_clips=max_num_clips_per_video,
                                               subsample_factor=video_subsample_factor)
    else:
        query_clip_sampler = make_clip_sampler(sampling_type="max",
                                               clip_length=video_clip_length,
                                               max_num_frame=max_num_sampled_frames_per_video,
                                               num_sampled_clips=_MAX_NUM_CLIPS,
                                               subsample_factor=video_subsample_factor)

    return UserCentricFewShotVideoClassificationDataset(
        video_database=video_database,
        category_sampler=object_sampler,
        video_instance_sampler=video_instance_sampler,
        support_video_clip_sampler=support_clip_sampler,
        query_video_clip_sampler=query_clip_sampler,
        num_episodes_per_user=num_episodes_per_user,
        user_cluster_label=use_object_cluster_labels,
        transform=transform,
        num_threads=num_threads,
        mode=mode)


if __name__ == '__main__':
    # self.normalize_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}  # imagenet mean train frame
    transform = torchvision.transforms.Compose([
        ApplyTransformToKey(
            key="support_image",
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(84),
                Div255(),
            ])
        ),
        ApplyTransformToKey(
            key="query_image",
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(84),
                Div255(),
            ])
        ),
        ApplyTransformToKey(
            key="episode_label",
            transform=torchvision.transforms.Compose([
                MutuallyExclusiveLabel(shuffle_ordered_label=True),
            ])
        ),
    ])

    datasets = ORBITUserCentricVideoFewShotClassification(data_path="/data02/datasets/ORBIT_microsoft",
                                                          num_episodes_per_user=10, transform=transform, num_threads=16,
                                                          mode="validation")
    dataloader = DataLoader(dataset=datasets, batch_size=1)
    for i, batch in enumerate(tqdm(dataloader)):
        a = 1
