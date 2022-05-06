from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from dataclasses import dataclass


@dataclass
class Instance:
    filename: str
    label: int


@dataclass
class VideoInstance(Instance):
    video_name: str


@dataclass
class UserCentricVideoInstance(Instance):
    """
    The fundamental data structure for ORBIT Video Few Shot Classification Dataset
    """
    video_name: str
    user_name: str
    category_name: str
    annotation_filename: str


@dataclass
class Episode:
    support_set: List[Instance]
    query_set: List[Instance]
