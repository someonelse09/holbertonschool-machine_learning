#!/usr/bin/env python3
"""This module includes implementation of One-Hot Decoder """
import numpy as np


def one_hot_decode(one_hot):
    """This function converts a one-hot
    matrix into a vector of labels"""
    if not isinstance(one_hot, np.ndarray):
        return None
    if one_hot.ndim != 2 or one_hot.size == 0:
        return None
    classes, m = one_hot.shape

    if not np.all((one_hot == 0) | (one_hot == 1)):
        return None
    labels = []

    for col_idx in range(m):
        column = one_hot[:, col_idx]
        count_of_ones = np.sum(column)
        if count_of_ones != 1:
            return None
        label_idx = np.argmax(column)
        labels.append(label_idx)
    return np.array(labels)
