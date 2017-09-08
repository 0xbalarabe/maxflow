import numpy as np
from scipy.linalg import norm


__author__ = 'balaogbeha'


def compute_relative_norm(A, x, b):
    norm_1 = norm(A*x - b)
    norm_2 = norm(b)

    return norm_1/norm_2


