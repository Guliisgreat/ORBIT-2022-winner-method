import random
import logging
from abc import ABC, abstractmethod
from fractions import Fraction
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union, List

import numpy as np

logger = logging.getLogger(__name__)


class Sampler(ABC):
    """
    """

    def __init__(self,
                 num_samples: int = None,
                 min_num_samples: int = None,
                 max_num_samples: int = None) -> None:
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
        Sample K (int) number of items from candidates

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
            # raise ValueError(f"the number of candidates: "
            #                  f"{len(candidates)} must be more than the number of samples from FixedSampler: {self.num_samples}")
            logger.warning(f"the number of candidates: "
                             f"{len(candidates)} must be more than the number of samples from FixedSampler: {self.num_samples}")
            return candidates
        return random.sample(candidates, k=self.num_samples)


class RandomSampler(Sampler):
    """
        Sample a random number of items from candidates
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
        Samples all candidates
    """
    def __call__(self, candidates: List) -> List:
        if self.max_num_samples:
            available_num_candidates = min(len(candidates), self.max_num_samples)
        else:
            available_num_candidates = len(candidates)
        return random.sample(candidates, k=available_num_candidates)


# class ChineseRestaurantProcessSampler(Sampler):
#
#     def __init__(self, max_num_samples: int = None,
#                  num_tables: int = 5,
#                  max_num_samples_per_table: int = 20,
#                  alpha: float = 0.5,
#                  theta: float = 1.0, ) -> None:
#         """
#
#             A sampler with Chinese Restaurant Process
#             Copy from Mengye Ren's OC-FewShot repo
#         """
#         super().__init__(max_num_samples)
#         self.num_tables = num_tables
#         self.max_num_samples_per_table = max_num_samples_per_table
#         self.alhpa = alpha
#         self.theta = theta
#
#     def __call__(self,
#                  candidates: List):
#         """
#          Example:
#              num_table=5, [0,1,2,3,4]
#              total_num_sample=10
#              candidates = ["a", "b", "c", "d", "e", "f", "g"]
#
#              Step 1: assign_samples_to_tables --> [0,1,1,1,4,4,2,0,2,3,0]
#              Step 2: map_table_id_to_candidates --> ["a":0, "b":1, "c":2, "d":3, "e":4]
#              Step 3: return sampled candidates --> ["a","b","b","b","e","e","c","a","f","a"]
#
#         """
#         table_ids = self._assign_people_to_tables(n=self.num_tables,
#                                                   alpha=self.alhpa,
#                                                   theta=self.theta,
#                                                   max_num=self.max_num_samples,
#                                                   max_num_per_cls=self.max_num_samples_per_table)
#         sampled_candidates = random.sample(candidates, k=self.num_tables)
#         table_id_to_candidate = {idx: candidate for idx, candidate in enumerate(sampled_candidates)}
#         return [table_id_to_candidate[table_id] for table_id in table_ids]
#
#     @staticmethod
#     def _assign_people_to_tables(
#             n,
#             alpha=0.5,
#             theta=1.0,
#             max_num=-1,
#             max_num_per_cls=20):
#         """.
#
#         Args:
#           n: Int. Number of tables.
#           alpha: Float. Discount parameter.
#           theta: Float. Strength parameter.
#           max_num: Int. Maximum number of people.
#           max_num_per_class: Int. Maximum number of people per table.
#         """
#         k = 0  # Current maximum table count.
#         c = 0  # Current sampled table.
#         m = 0  # Current total number of people.
#         m_array = np.zeros([n])
#         result = []
#         # We need n tables in total.
#         # print('alpha', alpha)
#         # print('max num', max_num)
#         # print('max num', max_num)
#         # print('max num pc', max_num_per_cls)
#         # print(max_num_per_cls)
#         while k <= n and (max_num < 0 or m < max_num):
#             # print('k', k, 'm', m)
#             p_new = (theta + k * alpha) / (m + theta)
#             if k == n:
#                 p_new = 0.0
#             # print('pnew', p_new)
#             p_old = (m_array[:k] - alpha) / (m + theta)
#             # print('pold', p_old)
#             pvals = list(p_old) + [p_new]
#             sample = np.random.multinomial(1, pvals, size=1)
#             sample = sample.reshape([-1])
#             # print('sample', sample, sample.shape)
#             if k == n:  # Reached the maximum classes.
#                 if sample[-1] == 1:  # Just sampled one more.
#                     break
#                     # continue
#
#             # print('sample', sample, sample.shape)
#             idx = np.argmax(sample)
#             if m_array[idx] < max_num_per_cls:
#                 m_array[idx] += 1
#             else:
#                 continue  # Cannot sample more on this table.
#             k = np.sum(m_array > 0)  # Update total table.
#             m += 1  # Update total guests.
#             result.append(idx)
#             # print(result)
#         # print(result)
#         # print('marray', m_array, 'results', result)
#         return result


def make_sampler(sampling_type: str,
                 num_samples: Optional[int] = None,
                 min_num_samples: int = 1,
                 max_num_samples: Optional[int] = None) -> Sampler:
    """
    Constructs the samplers found in ```` from the
    given arguments.

    Args:
        sampling_type (str): choose  sampler to return. It has three options:

            * fixed: constructs and return ``FixedSampler``
            * random: construct and return ``RandomSampler``
            * max: construct and return ``MaxSampler``

        num_samples:
        max_num_samples:
    """
    if sampling_type == "fixed":
        return FixedSampler(num_samples=num_samples, max_num_samples=max_num_samples)
    elif sampling_type == "random":
        return RandomSampler(num_samples=num_samples,  min_num_samples= min_num_samples, max_num_samples=max_num_samples)
    elif sampling_type == "max":
        return MaxSampler(num_samples=num_samples, max_num_samples=max_num_samples)
    else:
        raise NotImplementedError(f"{sampling_type} not supported")

