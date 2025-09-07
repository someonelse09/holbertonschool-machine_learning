#!/usr/bin/env pytyhon3
"""This module contains functions
to save and load Keras models"""

import tensorflow.keras as K


def save_model(network, filename):
    """Saves an entire model

    Args:
        network: the model to save
        filename: the path of the file that the model should be saved to

    Returns:
        None
    """
    network.save(filename)
    return None


def load_model(filename):
    """Loads an entire model

    Args:
        filename: the path of the file that the model should be loaded from

    Returns:
        the loaded model
    """
    return K.models.load_model(filename)
