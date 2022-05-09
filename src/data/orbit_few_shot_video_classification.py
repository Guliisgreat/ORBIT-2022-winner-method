import copy
import logging
import os
import json
import random
from pathlib import Path
import fnmatch
import re
import numpy as np
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
                          mode: str = "train") -> Dict:
    """
        Prepare ORBIT dataset metadata and save into one dictionary, which can be easily accessed by downstream
        torch.utils.data.Dataset

        Arguments:
            root_data_path (str): the root folder of orbit dataset
            mode (str): train, val, test
        Return:
            database (Nested Dict): the metadata of ORBIT dataset
                Format:
                user_name
                    * object_name
                        * video_type (clean or clutter)
                            * video_name
                                * user_name, category_name, category_label, category_cluster_label,
                                  video_type, filename_fullpath, annotation_fullpath
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

    cnt_video = 0
    logger.info("Prepare ORBIT dataset metadata... of {}_set".format(mode))
    for user_name in tqdm(user_names):
        video_database[user_name] = {}
        object_names = sorted(os.listdir(os.path.join(data_path, user_name)))
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
    logger.info("Done.")
    return video_database


class ORBITDatasetVideoInstanceSampler:
    """
        Given one object category and its video instance, we need to sample non-overlapped support instances
        and query instance from that object to construct an episode

        Requirements:
            1. At least one query instance sampled from each object
            2. At minimum 5 support instance sampled from each object

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
        Few shot video classification dataset for `ORBIT <https://github.com/microsoft/ORBIT-Dataset>`
        stored as image frames.

        This dataset handles randomly generating episodes via sampling object categories from each user and sampling
        support and query video instances from each object category. Also, this object handles sampling clips from
        each video sequence and loading clip frames. All io is done through :code:`iopath.common.file_io.PathManager`

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
                video_database (Dict[str, Dict]): the metadata of ORBIT dataset
                category_sampler (Optional[Sampler]): the sampler for object category
                video_instance_sampler (ORBITDatasetVideoInstanceSampler): the sampler for support and query video instances
                support_video_clip_sampler (ClipSampler): the sampler for support video clips
                query_video_clip_sampler (ClipSampler): the sampler for query video clips
                num_episodes_per_user: the number of randomly generated episodes for each user
                transform (Callable): transform functions and modules; Compatible with torchvision
                mode (str): train, validation, test
                num_threads (int): the number of CPU threads used to accelerate I/O; Load images (*.jpg) from the disk
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

        logger.info("Start randomly generate episodes in {}set".format(self.mode))
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
            Retrieves the next episode based on the video clip sampling strategy.

            Each episode has a support set, a query set, a user_id and a list of category names.

            In training, both the support set and the query set have multiple sampled video clips, and each of
            the clip has a 4-D clip_tensor, a label, image filenames and annotation of each frame

            In testing, the support set includes same elements, but the query set includes a list of
            length-invariance video sequences. Each video sequence has  a 4-D video_tensor, a label, image
            filenames and annotation of each frame

            Returns:
                A dictionary with the following format.

                .. code-block:: text
                    N = number of clips or videos; T = clip_length or video_length; C = 3, H = W = 224

                    {
                        'support_frames': <torch.FloatTensor>,  # Shape: (N, T, C, H, W)
                        'support_labels': <torch.LongTensor>,   # Shape: (N, )
                        'support_frame_filenames': <List[List[str]]>
                        'support_annotations': <List[List[Dict]]>
                        'query_frames': In train, <torch.FloatTensor>,  # Shape: (N, T, C, H, W)
                                        In test,  <List[torch.FloatTensor], # Shape: List[ (T, C, H, W)]
                        'query_targets": <torch.LongTensor>,   # Shape: (N, )
                        'query_frame_filenames': <List[List[str]]>
                        'query_annotations': <List[List[Dict]]>
                        'category_names': <List[str]>
                        "user_id": <str>
                      }
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

        # STEP 3: Apply Transform
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

        # STEP 4: In testing, we need to convert sampled query clips tensors into video tensors
        if self.mode != "train":
            query_frames, query_labels, query_frame_filenames = \
                self.__convert_multi_clips_into_video_sequences(query_frames, query_labels, query_frame_filenames)

        episode_dict = {"support_frames": support_frames,
                        "raw_support_frames": raw_support_frames,
                        "support_labels": support_labels,
                        "support_frame_filenames": support_frame_filenames,
                        "support_annotations": support_annotations,
                        "query_frames": query_frames,
                        "query_labels": query_labels,
                        "query_frame_filenames": query_frame_filenames,
                        "query_annotations": support_annotations,
                        "category_names": category_names,
                        "user_id": user_names
                        }

        # TODO: Hard coded for our proposed method, the edge detector need to access the raw RGB (0-255).
        #  Need to remove or refactor in the future
        if self.mode != "train":
            episode_dict["raw_support_frames"] = raw_support_frames

        return episode_dict

    def __len__(self):
        return len(self.episodes)

    def _generate_one_episode(self, category_names: List[str], category_name_to_instances: Dict[str, Dict]) -> Episode:
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
    def __shuffle_clips_indices(frames: torch.FloatTensor,
                                labels: torch.LongTensor,
                                frame_filenames: List,
                                frame_annotations: List) \
            -> Tuple[torch.FloatTensor, torch.LongTensor, List, List]:
        indices = np.random.permutation(len(labels))
        frames = frames[indices]
        labels = labels[indices]
        frame_filenames = [frame_filenames[idx] for idx in indices]
        frame_annotations = [frame_annotations[idx] for idx in indices]
        return frames, labels, frame_filenames, frame_annotations

    @staticmethod
    def __convert_multi_clips_into_video_sequences(frames: torch.FloatTensor,
                                                   labels: torch.LongTensor,
                                                   frame_filenames: List[List[str]]) \
            -> Tuple[List[Tensor], List[Any], List[Any]]:
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
        max_num_clips_per_video: int = 10,
        max_num_sampled_frames_per_video: int = 1000,
        video_subsample_factor: int = 1,
        video_clip_length: int = 8,
        use_object_cluster_labels: bool = False,
        num_episodes_per_user: int = 50,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        num_threads: int = 8,
) -> torch.utils.data.Dataset:
    """
        The refactored data pipeline for ORBIT dataset, sharing the same API with the original one.
        The original implementation can be found in https://github.com/microsoft/ORBIT-Dataset/blob/master/
        data/datasets.py

        This function is a wrapper of `UserCentricFewShotVideoClassificationDataset`

        Args:
            data_path (str): Path to ORBIT dataset root folder.

            mode (str): train, validation, test

            object_sampler (str): the type of sampler for objects; choice in ["random", "fixed", "max"]

            max_num_objects_per_user (int): the maximum number of sampled objects from each user

            support_instance_sampler (str): the type of sampler for support videos; choice in ["random", "fixed", "max"]

            query_instance_sampler (str): the type of sampler for query videos; choice in ["random", "fixed", "max"]

            support_num_shot (int): the number of videos sampled for the support set; (Only used in fixed sampler)

            query_num_shot (int): the number of videos sampled for the query set; (Only used in fixed sampler)

            max_num_instance_per_object (int): the maximum number of sampled videos from each object

            query_video_type (str): Determine where to sample the query set: "clean" or "clutter";
                In ORBIT_Challenge_2022, "clutter" is required

            support_clip_sampler (str): Determine how to sample non-overlapping clips from each support video sequence;
                choice in ["random", "fixed", "max", "uniform", "uniform_fixed_chunk_size"]

            query_clip_sampler (str): Determine how to sample non-overlapping clips from each query video
                sequence; In testing, "max" is strictly required to make sure all frames from each query videos are
                sampled.

            max_num_clips_per_video (int):  the maximum number of clips sampled from each video sequence

            max_num_sampled_frames_per_video (int): the maximum number of first frames used for sampling clips

            video_subsample_factor (int): Factor to subsample video frames before sampling clips.

            video_clip_length (int): the number of frames in each sampled video clip

            use_object_cluster_labels (bool): If True, use object cluster labels, otherwise use raw object labels.

            num_episodes_per_user (int): the number of episodes randomly generated from each user; In training, to be
                consistent with the number of total meta iterations in original ORBIT codebase, the default is 500
                (500 * 44 train users = 22k iterations); To introduce more diversity of episodes, we can further
                increase this number up to 1k. In testing, the default is 1.

            transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]: Compatible with transform functions
                in torchvision

            num_threads (int): the number of CPU threads used to accelerate I/O; Load images (*.jpg) from the disk
    """
    if mode not in ["train", "validation", "test"]:
        raise ValueError("ORBIT dataset has three subsets: train, validation, test; Please make a choice from them")

    if not os.path.exists(os.path.join(data_path, mode)):
        raise FileNotFoundError(os.path.join(data_path, mode))

    video_database = prepare_ORBIT_dataset(root_data_path=data_path, mode=mode)

    if object_sampler not in ["random", "max"]:
        raise NotImplementedError("Only supports 'random object sampler', 'max object sampler' so far")

    # Build object sampler
    object_sampler = make_sampler(sampling_type=object_sampler,
                                  min_num_samples=2,
                                  max_num_samples=max_num_objects_per_user)
    # Build video (instance) samplers
    support_instance_sampler = make_sampler(sampling_type=support_instance_sampler,
                                            num_samples=support_num_shot,
                                            max_num_samples=max_num_instance_per_object)
    query_instance_sampler = make_sampler(sampling_type=query_instance_sampler,
                                          num_samples=query_num_shot,
                                          max_num_samples=max_num_instance_per_object)
    video_instance_sampler = ORBITDatasetVideoInstanceSampler(support_instance_sampler=support_instance_sampler,
                                                              query_instance_sampler=query_instance_sampler,
                                                              query_video_type=query_video_type)

    # Build clip samplers
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
