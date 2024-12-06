import itertools
import numpy as np
from numpy import ndarray


def generate_next_vector(prev_vec: ndarray, radius: int) -> ndarray:
    """Algorithm for generating all vectors with positive entries of a given Manhattan length, specified in
    [1] J. M. Zhang and R. X. Dong, European Journal of Physics 31, 591 (2010)"""
    k = 0
    for num in range(len(prev_vec) - 2, -1, -1):
        if prev_vec[num] != 0:
            k = num
            break
    next_vec = np.zeros_like(prev_vec)
    next_vec[0:k] = prev_vec[0:k]
    next_vec[k] = prev_vec[k] - 1
    next_vec[k + 1] = radius - np.sum([next_vec[i] for i in range(k + 1)])
    return next_vec
