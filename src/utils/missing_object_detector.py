#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 16:55:51 2022

@author: huanliu
"""
from typing import List
import cv2
import numpy as np
import torch
from einops import rearrange, repeat


def select_frame_with_object(frame):
    '''
    frame: RGB image dim: (224, 224, 3)
    output: False --> non-object-exist frame
            True --> valid frame
    '''
    select_ratio = 0.0005
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # remove noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # CANNY
    canny = cv2.Canny(blur, threshold1=100, threshold2=200)
    ratio = sum(sum(canny / 255.0)) / 255.0 / 255.0
    if ratio <= select_ratio:
        return False
    else:
        return True


def filter_support_clips_without_object(support_raw_clips_frames: torch.FloatTensor,
                                        support_clips_frames: torch.FloatTensor,
                                        support_clips_labels: torch.LongTensor,
                                        support_clips_filenames: List[List[str]]):
    """
        support_clips_frames: [num_clips, 8, 3, 224, 224] M --> N; N < M
        support_clips_labels: [num_clips, ]

    """
    clip_length = 8
    clip_threshold_ratio = 0.5

    def process_single_clip(clip_frames):
        """
            Keep the clip --> True
        """
        frames = rearrange(clip_frames, "t c h w -> t h w c")
        valid_frame_flags = [select_frame_with_object(frame) for frame in frames]
        if sum(valid_frame_flags) < int(clip_length * clip_threshold_ratio):
            return False
        else:
            return True

    valid_clip_indices = \
        np.array([process_single_clip(clip_frames.cpu().numpy())
                  for clip_frames in support_raw_clips_frames]).nonzero()[0].tolist()
    support_clips_frames = support_clips_frames[valid_clip_indices, :, :, :, :]
    support_clips_labels = support_clips_labels[valid_clip_indices]
    support_clips_filenames = [filenames for idx, filenames in enumerate(support_clips_filenames) if
                               idx in valid_clip_indices]
    return support_clips_frames, support_clips_labels, support_clips_filenames
