#!/usr/bin/env python3
"""Trains a model using mini-batch gradient descent
with early stopping and learning rate decay"""
import tensorflow.keras as tf


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
    """Trains a model using mini-batch gradient descent

    Args:
        network: the model to train
        data: input data
        labels: labels of data
        batch_size: size of batch for mini-batch training
        epochs: number of passes through data
        validation_data: data to validate model with, if not None
        early_stopping: boolean indicating
        whether early stopping should be used
        patience: patience used for early stopping
        learning_rate_decay: boolean indicating whether
        learning rate decay should be used
        alpha: initial learning rate
        save_best: a boolean indicating whether
        to save the model after each epoch if it is the best
        filepath: the file path where the model should be saved
        decay_rate: decay rate
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

    if learning_rate_decay and validation_data is not None:
        def scheduler(epoch):
            """Learning rate scheduler using inverse time decay"""
            return alpha/(1 + decay_rate*epoch)
    lr_scheduler = K.callbacks.LearningRateScheduler(scheduler, verbose=1)
    callbacks.append(lr_scheduler)

    if save_best and filepath is not None:
        checkpoint = K.callbacks.ModelCheckpoint(
            file_name=filepath,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=0
        )
        callbacks.append(checkpoint)

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
