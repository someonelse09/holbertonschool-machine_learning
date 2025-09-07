#!/usr/bin/env python3
"""Converts a label vector into a one-hot matrix"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """This function converts a label vector into a one-hot matrix
    The last dimension of the one-hot matrix must be the number of classes"""
    return K.utils.to_categorical(labels, num_classes=classes)
