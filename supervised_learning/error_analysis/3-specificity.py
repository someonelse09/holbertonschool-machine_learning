#!/usr/bin/env python3
"""This module includes the function
that calculates the specificity
for each class in a confusion matrix"""

import numpy as np


def specificity(confusion):
    """
    Args:
        confusion is a confusion numpy.ndarray of shape
        (classes, classes) where row indices represent the correct
        labels and column indices represent the predicted labels
        classes is the number of classes
    Returns:
        a numpy.ndarray of shape (classes, )
        containing the specificity of each class
    """
    # specificity --> TN / (TN + FP)
    TP = np.diag(confusion)
    FN = np.sum(confusion, axis=1) - TP
    FP = np.sum(confusion, axis=0) - TP
    TN = np.sum(confusion) - (TP + FP + FN)
    return TN / (TN + FP)


