#!/usr/bin/env python3
"""This module includes the function
that calculates the precision for each
class in a confusion matrix"""

import numpy as np


def precision(confusion):
    """
    Args:
        confusion is a confusion numpy.ndarray of shape
         (classes, classes) where row indices represent the correct
          labels and column indices represent the predicted labels
        classes is the number of classes
    Returns:
        a numpy.ndarray of shape (classes,)
         containing the precision of each class
    """
    # precision --> TP / (TP + FP)
    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP
    return TP / (TP + FP)
