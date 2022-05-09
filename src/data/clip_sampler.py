import random
from abc import ABC, abstractmethod
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union, List
from numpy.lib.stride_tricks import sliding_window_view

from src.data.video import pad_last_frame


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
    Interface for clip samplers to sample multiple clips from a target video sequence,
    and returns a list of named-tuple ``ClipInfo``.
    """

    def __init__(self,
                 clip_length: int,
                 num_sampled_clips: int,
                 max_num_frame: int,
                 subsample_factor: int) -> None:
        """
            Args:
                clip_length (int): the number of frames in each sampled video clip
                num_sampled_clips (int): the number of video clips to be sampled
                max_num_frame (int): the first maximum number of frames will be access when sampling clips
                subsample_factor (int): the stride for frame sampling;
                    if the factor set to 1, all frames are sampled.
        """
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
        For each video sequence, we firstly split the video sequence evenly into multiple fix-sized clip candidates,
        then randomly sample a fixed number of clips from those clip candidates.
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
            stride (int): The stride of clip candidates; If `stride` = `clip_length`, clip candidates are non-overlapped.
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
                                      k=num_sampled_clips)
        sampled_clips.sort(key=lambda x: x[0])
        return sampled_clips


class RandomMultiClipSampler(FixedMultiClipSampler):
    """
        For each video sequence, we firstly randomly generate a number k to determine how many clips (k) would be sampled, and then select k
        clips from clip candidates.
    """
    def _sample_clips(self, non_overlapping_clips_candidates: List):
        num_sampled_clips = min(len(non_overlapping_clips_candidates), self.num_sampled_clips)
        random_num_clips = random.choice(range(1, num_sampled_clips + 1))
        sampled_clips = random.sample(non_overlapping_clips_candidates,
                                      k=random_num_clips)
        sampled_clips.sort(key=lambda x: x[0])
        return sampled_clips


class MaxMultiClipSampler(FixedMultiClipSampler):
    """
        For each video sequence, we select all clip candidates. It is used in sampling the query video sequences
        during testing, because all frames of one video must be evaluated.
    """
    def _sample_clips(self, non_overlapping_clips_candidates: List):
        num_sampled_clips = len(non_overlapping_clips_candidates)
        sampled_clips = random.sample(non_overlapping_clips_candidates,
                                      k=num_sampled_clips)
        sampled_clips.sort(key=lambda x: x[0])
        return sampled_clips


class UniformFixedNumberClipsMultiClipSampler(FixedMultiClipSampler):
    """
        To achieve high temporal coverage, we propose to uniformly sample K clips across each video sequence.
        We firstly split non-overlapped clip candidates evenly into K chunks, and then randomly select one clip from
        each chunk. The different video length of each video sequence can result in different chunk sizes.
        Thus, short videos have higher sampling rate.
    """
    def _sample_clips(self, non_overlapping_clips_candidates: List):
        num_sampled_clips = min(len(non_overlapping_clips_candidates), self.num_sampled_clips)
        chunk_size = len(non_overlapping_clips_candidates) // num_sampled_clips
        chunked_list = [non_overlapping_clips_candidates[i:i + chunk_size]
                        for i in range(0, len(non_overlapping_clips_candidates), chunk_size)]
        sampled_clips = [random.sample(chunk, k=1)[0] for chunk in chunked_list]
        sampled_clips.sort(key=lambda x: x[0])
        return sampled_clips


class UniformFixedChunkSizeMultiClipSampler(FixedMultiClipSampler):
    """
        To alleviate the above issue that video sequences have different sampling rates, we fix the chunk size when sampling. This sampling strategy allows the
        number of sampled clips is propotional to the video length. As a result, each video sequence can be sampled by the same sampling rate.
    """

    def _sample_clips(self, non_overlapping_clips_candidates: List):
        """
            chunk_size: the number of candidates in each chunk; One clip is sampled from each chunk.

        """
        # In ORBIT, 1000 / 8 = 125 candidates /10 = 12 candidates per chunk
        clip_length = len(non_overlapping_clips_candidates[0])
        chunk_size = self.max_num_frame // (self.num_sampled_clips * clip_length)
        if chunk_size < 1:
            raise ValueError("Each chunk must has at least one clip candidate")
        chunked_list = [non_overlapping_clips_candidates[i:i + chunk_size]
                        for i in range(0, len(non_overlapping_clips_candidates), chunk_size)]
        sampled_clips = [random.sample(chunk, k=1)[0] for chunk in chunked_list]
        sampled_clips.sort(key=lambda x: x[0])
        return sampled_clips


def make_clip_sampler(sampling_type: str, **kwargs) -> ClipSampler:
    """
    Construct the clip samplers found in ``src.data.clip_sampler`` from the
    given arguments.
    Args:
        sampling_type (str): spicify a particular clip sampler. There are five options:
            * fixed:
            * max:
            * random:
            * uniform_fixed_num_clips:
            * uniform_fixed_chunk_size:
        *kwargs: the args pass to the specified clip sampler constructor.
    """
    if sampling_type == "fixed":
        return FixedMultiClipSampler(**kwargs)
    elif sampling_type == "random":
        return RandomMultiClipSampler(**kwargs)
    elif sampling_type == "max":
        return MaxMultiClipSampler(**kwargs)
    elif sampling_type == "uniform_fixed_num_clips":
        return UniformFixedNumberClipsMultiClipSampler(**kwargs)
    elif sampling_type == "uniform_fixed_chunk_size":
        return UniformFixedChunkSizeMultiClipSampler(**kwargs)
    else:
        raise NotImplementedError(f"{sampling_type} not supported")
