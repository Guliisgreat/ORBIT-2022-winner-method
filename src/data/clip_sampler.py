import random
from abc import ABC, abstractmethod
from fractions import Fraction
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union, List

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import random
from abc import ABC, abstractmethod
from fractions import Fraction
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union, List

from numpy.lib.stride_tricks import sliding_window_view

from src.data.video import pad_last_frame


# class ClipInfoList(NamedTuple):
#     """
#     Named-tuple for clip information with:
#         clip_frame_indices  (List[int])
#         clip_end_sec (float): clip end time.
#         clip_index (int): clip index in the video.
#         aug_index (int): augmentation index for the clip. Different augmentation methods
#             might generate multiple views for the same clip.
#         is_last_clip (bool): a bool specifying whether there are more clips to be
#             sampled from the video.
#     """
#
#     clips_frame_index: List[int]
#     clip_end_index: List[int]
#     clip_subsample_factor: List[int]

class ClipInfo(NamedTuple):
    """
    Named-tuple for clip information with:
        clip_frame_indices  (List[int]): the clip's frame indices in the video.
        total_num_frames_video （int): the total number of frames in the video
        clip_subsample_factor (List[int]): the sampling rate of frames in the video
    """

    clip_frame_indices: List[int]
    total_num_frames_video: int
    clip_subsample_factor: int


class ClipSampler(ABC):
    """
    Interface for clip samplers that take a video time, previous sampled clip time,
    and returns a named-tuple ``ClipInfo``.
    """

    def __init__(self,
                 clip_length: int,
                 num_sampled_clips: int,
                 max_num_frame: int,
                 subsample_factor: int) -> None:
        self.clip_length = clip_length
        self.num_sampled_clips = num_sampled_clips
        self.max_num_frame = max_num_frame
        self.subsample_factor = subsample_factor

    @abstractmethod
    def __call__(
            self,
            total_num_frame_video: int,
    ) -> List[ClipInfo]:
        pass


class FixedMultiClipSampler(ClipSampler):
    """
    Evenly splits the video into clips of size clip_duration.
    """

    def __init__(
            self,
            clip_length: int,
            num_sampled_clips: int,
            max_num_frame: int,
            subsample_factor: int = 1,
            stride: Optional[int] = None,
    ):
        """
        Args:
            clip_duration (Union[float, Fraction]):
                The length of the clip to sample (in seconds).
            stride (floUnion[float, Fraction]at, optional):
                The amount of seconds to offset the next clip by
                default value of None is equivalent to no stride => stride == clip_duration.
        """
        super().__init__(clip_length=clip_length,
                         num_sampled_clips=num_sampled_clips,
                         max_num_frame=max_num_frame,
                         subsample_factor=subsample_factor)
        self._stride = stride if stride is not None else clip_length

        assert (
                self._stride > 0 and self._stride <= clip_length
        ), f"stride must be >0 and <= clip_duration ({clip_length})"

    def __call__(
            self, total_num_frame_video: int
    ) -> List[ClipInfo]:
        """
        Args:
            video_duration: (float): the duration of the video that's being sampled in seconds
        Returns:
            clip_info: (ClipInfo): includes the clip information (clip_start_time,

        """

        frame_indices = list(range(total_num_frame_video))[0:self.max_num_frame:self.subsample_factor]
        frame_indices = pad_last_frame(frame_indices, clip_length=self.clip_length)
        non_overlapping_clips_candidates = sliding_window_view(frame_indices, self.clip_length)[::self._stride]

        if len(non_overlapping_clips_candidates[-1]) != self.clip_length:
            raise ValueError("the number of elements from the last candidate {} != clip_length {}".
                             format(len(non_overlapping_clips_candidates[-1]), self.clip_length))

        sampled_clips = self._sample_clips(non_overlapping_clips_candidates.tolist())
        sampled_clips = [ClipInfo(clip_frame_indices=clip_indices,
                                  total_num_frames_video=len(frame_indices),
                                  clip_subsample_factor=self.subsample_factor)
                         for clip_indices in sampled_clips]
        return sampled_clips

    def _sample_clips(self, non_overlapping_clips_candidates: List):
        num_sampled_clips = min(len(non_overlapping_clips_candidates), self.num_sampled_clips)

        sampled_clips = random.sample(non_overlapping_clips_candidates,
                                      k=num_sampled_clips)  # each sampled_clip has its indices
        sampled_clips.sort(key=lambda x: x[0])
        return sampled_clips


class RandomMultiClipSampler(FixedMultiClipSampler):
    def _sample_clips(self, non_overlapping_clips_candidates: List):
        num_sampled_clips = min(len(non_overlapping_clips_candidates), self.num_sampled_clips)
        random_num_clips = random.choice(range(1, num_sampled_clips + 1))
        sampled_clips = random.sample(non_overlapping_clips_candidates,
                                      k=random_num_clips)  # each sampled_clip has its indices
        sampled_clips.sort(key=lambda x: x[0])
        return sampled_clips


class MaxMultiClipSampler(FixedMultiClipSampler):
    def _sample_clips(self, non_overlapping_clips_candidates: List):
        num_sampled_clips = len(non_overlapping_clips_candidates)
        sampled_clips = random.sample(non_overlapping_clips_candidates,
                                      k=num_sampled_clips)  # each sampled_clip has its indices
        sampled_clips.sort(key=lambda x: x[0])
        return sampled_clips


class FirstKMultiClipSampler(FixedMultiClipSampler):
    def _sample_clips(self, non_overlapping_clips_candidates: List):
        num_sampled_clips = min(len(non_overlapping_clips_candidates),self.num_sampled_clips)
        sampled_clips = non_overlapping_clips_candidates[:num_sampled_clips]
        sampled_clips.sort(key=lambda x: x[0])
        return sampled_clips


class UniformKMultiClipSampler(FixedMultiClipSampler):
    def _sample_clips(self, non_overlapping_clips_candidates: List):
        num_sampled_clips = min(len(non_overlapping_clips_candidates), self.num_sampled_clips)
        chunk_size = len(non_overlapping_clips_candidates) // num_sampled_clips
        chunked_list = [non_overlapping_clips_candidates[i:i + chunk_size]
                        for i in range(0, len(non_overlapping_clips_candidates), chunk_size)]
        sampled_clips = [random.sample(chunk, k=1)[0] for chunk in chunked_list]
        sampled_clips.sort(key=lambda x: x[0])
        return sampled_clips


class SkipFirstFramesMultiClipSampler(FixedMultiClipSampler):
    def _sample_clips(self, non_overlapping_clips_candidates: List):
        clip_length = len(non_overlapping_clips_candidates[0])
        num_skipped_frames = 30
        num_skipped_clips = num_skipped_frames // clip_length
        num_sampled_clips = min(len(non_overlapping_clips_candidates), self.num_sampled_clips)
        if num_sampled_clips <= (len(non_overlapping_clips_candidates) - num_skipped_clips):
            sampled_clips = random.sample(non_overlapping_clips_candidates[num_skipped_clips:],
                                          k=num_sampled_clips)  # each sampled_clip has its indices
        else:
            sampled_clips = random.sample(non_overlapping_clips_candidates,
                                          k=num_sampled_clips)  # each sampled_clip has its indices
        sampled_clips.sort(key=lambda x: x[0])
        return sampled_clips


class SkipFirstFramesUniformMultiClipSampler(FixedMultiClipSampler):
    def _sample_clips(self, non_overlapping_clips_candidates: List):
        clip_length = len(non_overlapping_clips_candidates[0])
        num_skipped_frames = 30
        num_skipped_clips = num_skipped_frames // clip_length
        num_sampled_clips = min(len(non_overlapping_clips_candidates), self.num_sampled_clips)
        if num_sampled_clips <= (len(non_overlapping_clips_candidates) - num_skipped_clips):
            non_overlapping_clips_candidates = non_overlapping_clips_candidates[num_skipped_clips:]
        chunk_size = len(non_overlapping_clips_candidates) // num_sampled_clips
        chunked_list = [non_overlapping_clips_candidates[i:i + chunk_size]
                        for i in range(0, len(non_overlapping_clips_candidates), chunk_size)]
        sampled_clips = [random.sample(chunk, k=1)[0] for chunk in chunked_list]
        sampled_clips.sort(key=lambda x: x[0])
        return sampled_clips


class UniformFixedRatioMultiClipSampler(FixedMultiClipSampler):
    def _sample_clips(self, non_overlapping_clips_candidates: List):
        ratio = 0.5
        num_sampled_clips = min(int(len(non_overlapping_clips_candidates) * ratio), self.num_sampled_clips)

        # 100 candidates, sample 10 --> hard threshold, ratio
        # 10 candidates, sample 2

        # num of samples propotional to length of video sequence
        chunk_size = len(non_overlapping_clips_candidates) // num_sampled_clips
        chunked_list = [non_overlapping_clips_candidates[i:i + chunk_size]
                        for i in range(0, len(non_overlapping_clips_candidates), chunk_size)]
        sampled_clips = [random.sample(chunk, k=1)[0] for chunk in chunked_list]
        sampled_clips.sort(key=lambda x: x[0])
        return sampled_clips


class UniformFixedChunkSizeMultiClipSampler(FixedMultiClipSampler):
    def _sample_clips(self, non_overlapping_clips_candidates: List):
        """
            chunk_size: the number of candidates in each chunk; One clip will be sampled from each chunk

        """
        # 1000 / 8 = 125 candidates /10 = 12 candidates per chunk
        clip_length = len(non_overlapping_clips_candidates[0])
        chunk_size = 1000 // (self.num_sampled_clips * clip_length)
        if chunk_size < 1:
            raise ValueError("Each chunk must has at least one clip candidate")
        chunked_list = [non_overlapping_clips_candidates[i:i + chunk_size]
                        for i in range(0, len(non_overlapping_clips_candidates), chunk_size)]
        sampled_clips = [random.sample(chunk, k=1)[0] for chunk in chunked_list]
        sampled_clips.sort(key=lambda x: x[0])
        return sampled_clips


def make_clip_sampler(sampling_type: str, **kargs) -> ClipSampler:
    """
    Constructs the clip samplers found in ``pytorchvideo.data.clip_sampling`` from the
    given arguments.
    Args:
        sampling_type (str): choose clip sampler to return. It has three options:
            * max:
            * random:

        *args: the args to pass to the chosen clip sampler constructor.
    """
    if sampling_type == "fixed":
        return FixedMultiClipSampler(**kargs)
    elif sampling_type == "random":
        return RandomMultiClipSampler(**kargs)
    elif sampling_type == "max":
        return MaxMultiClipSampler(**kargs)
    elif sampling_type == "first":
        return FirstKMultiClipSampler(**kargs)
    elif sampling_type == "uniform":
        return UniformKMultiClipSampler(**kargs)
    elif sampling_type == "uniform_fixed_chunk_size":
        return UniformFixedChunkSizeMultiClipSampler(**kargs)
    elif sampling_type == "skip_first":
        return SkipFirstFramesMultiClipSampler(**kargs)
    elif sampling_type == "skip_uniform":
        return SkipFirstFramesUniformMultiClipSampler(**kargs)
    elif sampling_type == "uniform_fixed_ratio":
        return UniformFixedRatioMultiClipSampler(**kargs)
    else:
        raise NotImplementedError(f"{sampling_type} not supported")
