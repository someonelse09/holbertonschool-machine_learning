#!/usr/bin/env python3
"""This module includes the
function to train the model"""
import tensorflow.keras as K

def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, verbose=True, shuffle=False):
    """Trains a model using mini-batch gradient descent

    Args:
        network: the model to train
        data: input data
        labels: labels of data
        batch_size: size of batch for mini-batch training
        epochs: number of passes through data
        validation_data: data to validate model with, if not None
        verbose: whether to print progress during training
        shuffle: whether to shuffle batches between epochs

    Returns:
        The History object generated after training the model"""
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data
    )
    return history
