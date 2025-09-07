#!/usr/bin/env python3
"""This module includes testing the model"""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network

    Args:
        network: the model to test
        data: input data to test the model with
        labels: the correct one-hot labels of data
        verbose: whether to print progress during testing

    Returns:
        The loss and accuracy of the model with the testing data, respectively
    """
    return network.evaluate(data, labels, verbose=verbose)
