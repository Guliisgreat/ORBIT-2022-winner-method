import torch
from typing import Tuple


def mutually_exclusive_label(
        episode_global_support_labels: torch.LongTensor,
        episode_global_query_labels: torch.LongTensor,
        shuffle_ordered_label: bool = False,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    Convert the globel labels across the dataset into the each sampled episode's local label
    Attention: local labels need to follow the ascending order of global labels


    E.g. A 5-way episode whose labels are [89, 12, 118, 221, 435, 89, 12, 118, 221, 435]
         --> episode targets [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

    """
    device = episode_global_support_labels.device
    unique_global_support_labels = episode_global_support_labels.unique()
    unique_global_query_labels = episode_global_query_labels.unique()
    if not torch.equal(unique_global_support_labels, unique_global_query_labels):
        raise ValueError("Different gt labels between the support and the query. "
                         "gt label in support set = {}, gt label in query set = {}".
                         format(unique_global_support_labels, unique_global_query_labels))
    if shuffle_ordered_label:
        unique_global_support_labels = unique_global_support_labels[
            torch.randperm(unique_global_support_labels.nelement())]

    global_to_local_mapping = {global_label.item(): idx for idx, global_label in
                               enumerate(unique_global_support_labels)}

    def map_global_to_local(global_labels):
        episode_local_labels = []
        for global_label in global_labels:
            episode_local_labels.append(global_to_local_mapping[global_label.item()])
        return episode_local_labels

    support_local_labels = map_global_to_local(episode_global_support_labels)
    query_local_labels = map_global_to_local(episode_global_query_labels)

    return torch.LongTensor(support_local_labels, device=device), torch.LongTensor(query_local_labels, device=device)


def div_255(x: torch.Tensor) -> torch.Tensor:
    """
    Divide the given tensor x by 255.
    Args:
        x (torch.Tensor): The input tensor.
    Returns:
        y (torch.Tensor): Scaled tensor by dividing 255.
    """
    y = x / 255.0
    return y