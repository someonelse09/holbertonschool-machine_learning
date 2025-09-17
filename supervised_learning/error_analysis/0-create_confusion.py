#!/usr/bin/env python3
"""This module includes the
function that creates a confusion matrix"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Args:
        labels is a one-hot numpy.ndarray of shape
            (m, classes) containing the correct labels for each data point
        logits is a one-hot numpy.ndarray of shape
            (m, classes) containing the predicted labels
        m is the number of data points
        classes is the number of classes

    Returns:
        a confusion numpy.ndarray of shape (classes, classes)
            with row indices representing the correct labels and
                column indices representing the predicted labels
    """
    classes = labels.shape[1]

    true_classes = np.argmax(labels, axis=1)
    predicted_classes = np.argmax(logits, axis=1)

    confusion_matrix = np.zeros((classes, classes), dtype=int)
    for t, p in zip(true_classes, predicted_classes):
        confusion_matrix[t, p] += 1
    return confusion_matrix
