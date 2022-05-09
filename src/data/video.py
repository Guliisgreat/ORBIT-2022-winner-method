import logging
import os
import re
from typing import Callable, Dict, List, Optional, Tuple
from pathlib import Path
from einops import rearrange
import numpy as np
import torch

from .utils import _load_images_with_retries


logger = logging.getLogger(__name__)


def pad_last_frame(frames_path: List[str], clip_length: int = 8):
    """
        If not divisible by clip_length, pad with last frame until it is
        Follow https://github.com/microsoft/ORBIT-Dataset/blob/5a2b4e852d610528403f12a5130f676e5c6e48bc/data/datasets.py#L328
    """
    spare_frames = len(frames_path) % clip_length
    if spare_frames > 0:
        frames_path.extend([frames_path[-1]] * (clip_length - spare_frames))
    return frames_path


class FrameVideo:
    """
        FrameVideo is an abstractions for accessing clips based on their frame indices in a video where each frame
        is stored as an image. PathManager is used for frame reading.

    """

    def __init__(
            self,
            video_folder_path: str,
            num_threads: int = 8,
            clip_length: int = None,
    ) -> None:
        """
            Args:
                video_folder_path (str): the fullpath of the video where each frame is store as an image (*.jpg)
                clip_length (int): the number of frames in each clip
                num_threads (int): the number of CPU threads used for I/O frame image loading
        """
        self.video_folder_path = video_folder_path
        self.num_threads = num_threads
        self.clip_length = clip_length
        self._duration = len(os.listdir(video_folder_path))
        self.video_image_paths = [os.path.join(video_folder_path, image_filename) for image_filename in
                                  os.listdir(video_folder_path)]
        self.video_image_paths.sort(key=lambda x: re.findall(r'\d+', Path(x).name)[-1])

    @property
    def name(self) -> str:
        return Path(self.video_folder_path).name

    @property
    def total_num_frames(self) -> int:
        """
        Returns:
            duration: the video's duration/end-time in seconds.
        """
        return self._duration

    def get_single_clip(self,
                        clip_frame_indices: List[int],
                        ) -> Tuple[torch.Tensor, List]:
        """
            Get one clip's frame tensors and its frame paths from the clip indices

            Args:
                clip_frame_indices (List[int]): the index of each frame in the clip

            Returns:
                image_tensor (torch.FloatTensor): the clip's frame tensors
                image_paths (List[str]): the clip's frame filenames
        """
        if clip_frame_indices[0] < 0 or clip_frame_indices[-1] > self._duration:
            logger.warning(
                f"No frames found within {clip_frame_indices[0]} and {clip_frame_indices[-1]} seconds. Video starts"
                f"at time 0 and ends at {self._duration}."
            )
            raise NotImplementedError
        if not self.clip_length:
            raise ValueError("Require 'clip_length‘ to get clip(s) from the video sequence")
        image_paths = [self.video_image_paths[idx] for idx in clip_frame_indices]
        images_tensor = _load_images_with_retries(image_paths=image_paths, num_threads=self.num_threads)

        return images_tensor, image_paths

    def get_multiple_clips(self, clips_frame_indices_list: List[List[int]]) -> Tuple[torch.Tensor, List[List]]:
        """
            Get multiple clip's frame tensors and their paths from the a list of clip indices

            Args:
                clips_frame_indices_list (List[List[int]]): the index of each frame in each clip

            Returns:
                image_tensor (torch.FloatTensor): multiple clips' frame tensors  # Shape = [n, t, h, w, c]
                    n = num_clips, t=clip_length
                image_paths (List[List[str]]): multiple clips' frame filenames
        """
        if not self.clip_length:
            raise ValueError("Require 'clip_length‘ to get clip(s) from the video sequence")

        image_paths = []
        for clip_indices in clips_frame_indices_list:
            if len(clip_indices) != self.clip_length:
                raise ValueError("Error! Each sampled clip must have same length = {}".format(self.clip_length))
            image_paths.extend([self.video_image_paths[idx] for idx in clip_indices])
        images_tensor = _load_images_with_retries(image_paths=image_paths, num_threads=self.num_threads)
        images_tensor = rearrange(images_tensor, "(n t) h w c -> n t h w c", t=self.clip_length)
        image_paths = rearrange(np.array(image_paths), "(n t) -> n t", t=self.clip_length).tolist()
        return images_tensor, image_paths
