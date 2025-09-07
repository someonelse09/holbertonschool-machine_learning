#!/usr/bin/env python3
"""This odule includes functions
for saving and loading the weights"""

import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """
    Saves a model’s weights

    Args:
        network: the model whose weights should be saved
        filename: the path of the file that the weights should be saved to
        save_format: the format in which the weights should be saved

    Returns: None
    """
    return network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """
    Loads a model’s weights

    Args:
        network: the model to which the weights should be loaded
        filename: the path of the file that the weights should be loaded from

    Returns: None
    """
    return network.load_weights(filename)
