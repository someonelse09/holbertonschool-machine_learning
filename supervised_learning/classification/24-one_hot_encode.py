#!/usr/bin/env python3
"""This module includes implementation of One-Hot Encoder """
import numpy as np


def one_hot_encode(Y, classes):
    """This function converts a numeric
    label vector into a one-hot matrix"""
    if not isinstance(Y, np.ndarray) and type(classes) is not int:
        return None
    if len(Y) == 0:
        return None
    if classes <= 0:
        return None
    m = len(Y)
    oneh_matrix = np.zeros((classes, m), dtype=float)
    i = 0
    for k in Y:
        if k < 0 or k >= classes:
            return None
        oneh_matrix[k][i] = 1.0
        i += 1

    return oneh_matrix
