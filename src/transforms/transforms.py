from typing import Callable, Dict, List, Optional, Tuple
import torch
import torchvision

from src.transforms.functional import div_255, mutually_exclusive_label


class ApplyTransformToKey:
    """
    Applies transform_moduel to key of dictionary input.
    Args:
        key (str): the dictionary key the transform_moduel is applied to
        transform (callable): the transform_moduel that is applied
    # Example:
    #     >>>   transforms.ApplyTransformToKey(
    #     >>>       key='image',
    #     >>>       transform_moduel=UniformTemporalSubsample(num_video_samples),
    #     >>>   )
    # """

    def __init__(self, key: str, transform: Callable):
        self._key = key
        self._transform = transform

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x[self._key] = self._transform(x[self._key])
        return x


class MutuallyExclusiveLabel(torch.nn.Module):
    """

    """

    def __init__(self, shuffle_ordered_label):
        super().__init__()
        self.shuffle_ordered_label = shuffle_ordered_label

    def forward(self, episode_global_labels: Tuple[torch.LongTensor, torch.LongTensor]):
        """
        Args:
            episode_global_labels = (episode_global_support_labels, episode_global_query_labels)

        """
        episode_global_support_labels, episode_global_query_labels = episode_global_labels
        return mutually_exclusive_label(episode_global_support_labels,
                                        episode_global_query_labels,
                                        self.shuffle_ordered_label)


class Div255(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.div_255``.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scale clip frames from [0, 255] to [0, 1].
        Args:
            x (Tensor): A tensor of the clip's RGB frames with shape:
                (C, T, H, W).
        Returns:
            x (Tensor): Scaled tensor by dividing 255.
        """

        return torchvision.transforms.Lambda(div_255)(x)
