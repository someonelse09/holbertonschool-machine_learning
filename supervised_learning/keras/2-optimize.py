#!/usr/bin/env python3
"""Optimizes the neural network using the Keras library"""

import tensorflow as tf


def optimize_model(network, alpha, beta1, beta2):
    """This function sets up Adam optimization
    for a keras model with categorical crossentropy
    loss and accuracy metrics"""
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1
        beta_2=beta2
    )
    network.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
