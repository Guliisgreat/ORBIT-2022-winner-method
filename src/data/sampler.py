import random
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union, List


logger = logging.getLogger(__name__)


class Sampler(ABC):
    """
        Interface for generic samplers to sample items from candidates, and returns a list of sampled items.
    """

    def __init__(self,
                 num_samples: int = None,
                 min_num_samples: int = None,
                 max_num_samples: int = None) -> None:
        """
            Args:
                num_samples (int): the number of samples
                min_num_samples (int): the minimum number of samples
                max_num_samples (int): the maximum number of samples
        """
        if max_num_samples and num_samples and num_samples > max_num_samples:
            raise ValueError("the number of samples: {} from the sampler must be less than its max_limit: {}".
                             format(num_samples, max_num_samples))
        self.num_samples = num_samples
        self.min_num_samples = min_num_samples
        self.max_num_samples = max_num_samples

    @abstractmethod
    def __call__(self, candidates: List) -> List:
        pass


class FixedSampler(Sampler):
    """
        Sample K number of items from candidates

        Strict Requirements:
            num_samples <= num_candidates
            num_samples <= max_num_samples

    """

    def __init__(self, num_samples: int = None, max_num_samples: int = None) -> None:
        super().__init__(num_samples, max_num_samples)
        if not num_samples:
            raise ValueError("FixedSampler must have a fixed number of samples")

    def __call__(self, candidates: List) -> List:
        if len(candidates) < self.num_samples:
            logger.warning(f"the number of candidates: "
                           f"{len(candidates)} must be more than the number of samples from FixedSampler:"
                           f" {self.num_samples}")
            return candidates
        return random.sample(candidates, k=self.num_samples)


class RandomSampler(Sampler):
    """
        Sample a random number of items from candidates

        Requirement:
            min_num_sample <= num_samples <= num_candidates
    """

    def __call__(self, candidates: List) -> List:
        if self.max_num_samples:
            available_num_candidates = min(len(candidates), self.max_num_samples)
        else:
            available_num_candidates = len(candidates)
        k = random.choice(range(self.min_num_samples, available_num_candidates + 1))
        return random.sample(candidates, k=k)


class MaxSampler(Sampler):
    """
        Sample all candidates

    """

    def __call__(self, candidates: List) -> List:
        if self.max_num_samples:
            available_num_candidates = min(len(candidates), self.max_num_samples)
        else:
            available_num_candidates = len(candidates)
        return random.sample(candidates, k=available_num_candidates)


class ChineseRestaurantProcessSampler(Sampler):

    def __init__(self, max_num_samples: int = None,
                 num_tables: int = 5,
                 max_num_samples_per_table: int = 20,
                 alpha: float = 0.5,
                 theta: float = 1.0, ) -> None:
        """

            Sample with Chinese Restaurant Process
        """
        super().__init__(max_num_samples)
        pass

    def __call__(self,
                 candidates: List):
        """
         Example:
             num_table=5, [0,1,2,3,4]
             total_num_sample=10
             candidates = ["a", "b", "c", "d", "e", "f", "g"]

             Step 1: assign_samples_to_tables --> [0,1,1,1,4,4,2,0,2,3,0]
             Step 2: map_table_id_to_candidates --> ["a":0, "b":1, "c":2, "d":3, "e":4]
             Step 3: return sampled candidates --> ["a","b","b","b","e","e","c","a","f","a"]

        """
        pass


def make_sampler(sampling_type: str,
                 num_samples: Optional[int] = None,
                 min_num_samples: int = 1,
                 max_num_samples: Optional[int] = None) -> Sampler:
    """
    Build the samplers in ``src.data.sampler`` by the
    given arguments.

    Args:
        sampling_type (str): specify a sampler to return. There are three options:
            * fixed: constructs and return ``FixedSampler``
            * random: construct and return ``RandomSampler``
            * max: construct and return ``MaxSampler``
        num_samples (int): the number of samples
        min_num_samples (int): the minimum number of samples
        max_num_samples (int): the maximum number of samples
    """
    if sampling_type == "fixed":
        return FixedSampler(num_samples=num_samples, max_num_samples=max_num_samples)
    elif sampling_type == "random":
        return RandomSampler(num_samples=num_samples, min_num_samples=min_num_samples, max_num_samples=max_num_samples)
    elif sampling_type == "max":
        return MaxSampler(num_samples=num_samples, max_num_samples=max_num_samples)
    else:
        raise NotImplementedError(f"{sampling_type} not supported")
