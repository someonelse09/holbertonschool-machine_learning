#!/usr/bin/env python3
"""This module includes the function that
calculates the sensitivity for each
class in a confusion matrix"""

import numpy as np


def sensitivity(confusion):
    """
    Args:
        confusion is a confusion numpy.ndarray of shape
           (classes, classes) where row indices represent the
              correct labels and column indices represent the predicted labels
        classes is the number of classes
    Returns:
        a numpy.ndarray of shape (classes,)
            containing the sensitivity of each class
    """
    # sensitivity --> TP / (TP + FN)
    TP = np.diag(confusion)
    FN = np.sum(confusion, axis=1) - TP
    sensitivity_values = TP / (TP + FN)
    return sensitivity_values
