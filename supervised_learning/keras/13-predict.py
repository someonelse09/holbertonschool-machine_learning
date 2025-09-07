#!/usr/bin/env python3
"""This module contains function to return predictions"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a neural network

    Args:
        network: the model to make the prediction with
        data: the input data to make the prediction with
        verbose: whether to print progress during prediction

    Returns:
        The prediction for the data
    """
    return network.predict(data, verbose=verbose)
