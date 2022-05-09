import threading
import logging
import time
from iopath.common.file_io import g_pathmgr
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

import numpy as np
import cv2
import torch
import concurrent.futures

logger = logging.getLogger(__name__)

"""
    Modified on https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/data/utils.py

"""


def thwc_to_cthw(data: torch.Tensor) -> torch.Tensor:
    """
    Permute tensor from (time, height, weight, channel) to
    (channel, height, width, time).
    """
    return data.permute(3, 0, 1, 2)


def thwc_to_tchw(data: torch.Tensor) -> torch.Tensor:
    """
    Permute tensor from (time, height, weight, channel) to
    (time, channel, height, width).
    """
    return data.permute(0, 3, 1, 2)


def thwc_to_cthw_numpy(data: np.ndarray) -> np.ndarray:
    """
    Permute tensor from (time, height, weight, channel) to
    (channel, height, width, time).
    """
    return data.transpose(3, 0, 1, 2)


def optional_threaded_foreach(
        target: Callable, args_iterable: Iterable[Tuple], multithreaded: bool,
):
    """
    Applies 'target' function to each Tuple args in 'args_iterable'.
    If 'multithreaded' a thread is spawned for each function application.

    Args:
        target (Callable):
            A function that takes as input the parameters in each args_iterable Tuple.

        args_iterable (Iterable[Tuple]):
            An iterable of the tuples each containing a set of parameters to pass to
            target.

        multithreaded (bool):
            Whether or not the target applications are parallelized by thread.


    """

    if multithreaded:
        threads = []
        for args in args_iterable:
            thread = threading.Thread(target=target, args=args)
            thread.start()
            threads.append(thread)

        for t in threads:  # Wait for all threads to complete
            t.join()
    else:
        for args in args_iterable:
            target(*args)


def optional_multi_thread(target, args, max_workers=8):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        [executor.submit(target, arg[0], arg[1]) for arg in args]


def _load_images_with_retries(
        image_paths: List[str],
        num_retries: int = 10,
        num_threads: int = 8,
        rgb: bool = True
) -> torch.Tensor:
    """
    Loads the given image paths using PathManager, decodes them as RGB images and
    returns them as a stacked tensors.
    Args:
        image_paths (List[str]): a list of paths to images.
        num_retries (int): number of times to retry image reading to handle transient error.
        num_threads (int): if images are fetched via multiple threads in parallel.
        rgb (bool): fetch RGB img or Greyscale
    Returns:
        A tensor of the clip's RGB frames with shape:
        (time, height, width, channel). The frames are of type torch.uint8 and
        in the range [0 - 255]. Raises an exception if unable to load images.
    """
    imgs = [None for i in image_paths]

    def fetch_image(image_index: int, image_path: str) -> None:
        for i in range(num_retries):
            with g_pathmgr.open(image_path, "rb") as f:
                img_str = np.frombuffer(f.read(), np.uint8)
                if rgb:
                    img_bgr = cv2.imdecode(img_str, flags=cv2.IMREAD_COLOR)
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                else:
                    img_rgb = cv2.imdecode(img_str, cv2.IMREAD_GRAYSCALE)
                    img_rgb = np.expand_dims(img_rgb, axis=2)

            if img_rgb is not None:
                imgs[image_index] = img_rgb
                return
            else:
                logging.warning(f"Reading attempt {i}/{num_retries} failed.")
                time.sleep(1e-6)

    optional_multi_thread(fetch_image, list(enumerate(image_paths)), max_workers=num_threads)

    for img, p in zip(imgs, image_paths):
        if img is None:
            raise Exception("Failed to load images from {}".format(p))

    return torch.as_tensor(np.stack(imgs))


def _flatten_nested_list(nested_list):
    return [item for sublist in nested_list for item in sublist]


def sort_dictionary_with_key(mapping):
    mapping = {k: v for k, v in sorted(mapping.items(), key=lambda item: item[0])}
    return mapping


def sort_dictionary_with_value(mapping):
    mapping = {k: v for k, v in sorted(mapping.items(), key=lambda item: item[1])}
    return mapping


def _ndarray_to_list(elements):
    return [element for element in elements]
