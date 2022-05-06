import logging
import math
import os
import re
import time
from typing import Callable, Dict, List, Optional, Tuple
from pathlib import Path
from einops import rearrange
import numpy as np

import torch

from .utils import _load_images_with_retries


logger = logging.getLogger(__name__)


def pad_last_frame(frames_path: List[str], clip_length: int = 8):
    """
        if not divisible by clip_length, pad with last frame until it is
    """
    spare_frames = len(frames_path) % clip_length
    if spare_frames > 0:
        frames_path.extend([frames_path[-1]] * (clip_length - spare_frames))
    return frames_path


class FrameVideo:
    """
    FrameVideo is an abstractions for accessing clips based on their start and end
    time for a video where each frame is stored as an image. PathManager is used for
    frame image reading, allowing non-local uri's to be used.
    """

    def __init__(
            self,
            video_folder_path: str,
            num_threads: int = 8,
            max_total_num_frame: int = 1000,
            clip_length: int = None,
    ) -> None:
        self.video_folder_path = video_folder_path
        self.num_threads = num_threads
        self.max_total_num_frame = max_total_num_frame
        self.clip_length = clip_length
        self._duration = len(os.listdir(video_folder_path))
        self.video_image_paths = [os.path.join(video_folder_path, image_filename) for image_filename in
                                  os.listdir(video_folder_path)]
        # Example: /train/P697/wallet/clean/P697--wallet--clean--4zbWUtNoJTEgJ8nddFyqXH6M6U7MglvklavVflYeVkU
        # /P697--wallet--clean--4zbWUtNoJTEgJ8nddFyqXH6M6U7MglvklavVflYeVkU-00003.jpg
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
        if clip_frame_indices[0] < 0 or clip_frame_indices[-1] > self._duration:
            logger.warning(
                f"No frames found within {clip_frame_indices[0]} and {clip_frame_indices[-1]} seconds. Video starts"
                f"at time 0 and ends at {self._duration}."
            )
            raise NotImplementedError
        if not self.clip_length:
            raise ValueError("Require 'clip_length‘ to get clip(s) from the video sequence")
        clip_image_paths = [self.video_image_paths[idx] for idx in clip_frame_indices]
        images_tensor = _load_images_with_retries(image_paths=clip_image_paths, num_threads=self.num_threads)

        return images_tensor, clip_image_paths

    def get_multiple_clips(self, clips_frame_indices_list: List[List[int]]) -> Tuple[torch.Tensor, List[List]]:
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
