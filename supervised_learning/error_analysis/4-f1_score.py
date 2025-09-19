#!/usr/bin/env python3
"""This module includes the function
that calculates the F1 score
of the confusion matrix"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Args:
        confusion is a confusion numpy.ndarray of shape
        (classes, classes) where row indices represent the
        correct labels and column indices represent the predicted labels
        classes is the number of classes
    Returns:
        a numpy.ndarray of shape (classes, )
        containing the F1 score of each class
    """
    # In this context the terms recall and sensitivity are interchangeable
    # f1_score --> 2 * Precision * Sensitivity / ( Precision + Sensitivity )
    # Or f1_score is harmonic mean of Precision and Sensitivity
    sy = sensitivity(confusion)
    precs = precision(confusion)
    return 2 * sy * precs / (precs + sy)
