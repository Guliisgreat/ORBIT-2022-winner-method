#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 16:55:51 2022

@author: huanliu
"""
from typing import List, Tuple
import cv2
import numpy as np
import torch
from einops import rearrange, repeat


def select_frame_with_object(frame: np.ndarray, select_ratio: float = 0.0005) -> bool:
    """
        Determine whether the frame has non-object-present issue using Canny Edge detector to

        Args:
            frame (numpy.ndarray): RGB image  # Shape  [224, 224, 3]
            select_ratio (float): the hand-tuned ratio to determine non-object-present issue

        Returns:
            flag (bool): False --> the frame has non-object-present issue.
                         True --> the frame is valid
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # remove noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # CANNY
    canny = cv2.Canny(blur, threshold1=100, threshold2=200)
    ratio = sum(sum(canny / 255.0)) / 255.0 / 255.0
    flag = False if ratio <= select_ratio else True
    return flag


def filter_support_clips_without_object(support_raw_clips_frames: torch.FloatTensor,
                                        support_clips_frames: torch.FloatTensor,
                                        support_clips_labels: torch.LongTensor,
                                        support_clips_filenames: List[List[str]],
                                        clip_length: int = 8,
                                        clip_threshold_ratio: float = 0.5) \
        -> Tuple[torch.FloatTensor, torch.FloatTensor, List[List[str]]]:
    """
        Determine which sampled support clips has non-object-present issue and remove them
        Rule: In one clip, if the number of frames with non-object-present issues is more than the threshold, that
            clip will be removed from the support set.

        Args:
            support_clips_frames (torch.FloatTensor): multiple support clips frame tensors  # Shape = [N, T, C, H, W]
            support_raw_clips_frames (torch.FloatTensor): multiple support clips raw frame tensors
                Shape = [N, T, C, H, W]; the raw frame means without normalization, pixel value between 0 ~ 255,
                because the edge detector only accept the raw image array
            support_clips_labels (torch.LongTensor): multiple support clips' labels  # Shape = [N, ]
            support_clips_filenames (List[List[str]]): multiple support clips' filenames
            clip_length (int): the number of frames in each support clip
            clip_threshold_ratio (float): the threshold to determine whether the clip is valid

        Returns:
            support_clips_frames (torch.FloatTensor): multiple valid support clips frame tensors
                Shape = [M, T, C, H, W]. Here, M <= N
            support_clips_labels (torch.LongTensor): multiple valid support clips' labels  # Shape = [M, ]
            support_clips_filenames (List[List[str]]): multiple valid support clips' filenames

    """
    def process_single_clip(clip_frames):
        frames = rearrange(clip_frames, "t c h w -> t h w c")
        valid_frame_flags = [select_frame_with_object(frame) for frame in frames]
        return False if sum(valid_frame_flags) < int(clip_length * clip_threshold_ratio) else True

    valid_clip_indices = \
        np.array([process_single_clip(clip_frames.cpu().numpy())
                  for clip_frames in support_raw_clips_frames]).nonzero()[0].tolist()
    support_clips_frames = support_clips_frames[valid_clip_indices, :, :, :, :]
    support_clips_labels = support_clips_labels[valid_clip_indices]
    support_clips_filenames = [filenames for idx, filenames in enumerate(support_clips_filenames) if
                               idx in valid_clip_indices]
    return support_clips_frames, support_clips_labels, support_clips_filenames
