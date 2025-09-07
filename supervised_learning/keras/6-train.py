#!/usr/bin/env python3
"""Converts a label vector into a one-hot matrix"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """Trains a model using mini-batch gradient descent

    Args:
        network: the model to train
        data: input data
        labels: labels of data
        batch_size: size of batch for mini-batch training
        epochs: number of passes through data
        validation_data: data to validate model with, if not None
        early_stopping: boolean indicating whether early stopping should be used
        patience: patience used for early stopping
        verbose: whether to print progress during training
        shuffle: whether to shuffle batches between epochs

    Returns:
        The History object generated after training the model
    """
    callbacks = []
    if early_stopping and validation_data is not None:
        early_stopping_callback = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )
        callbacks.append(early_stopping_callback)
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callbacks if callbacks else None
    )
    return history
