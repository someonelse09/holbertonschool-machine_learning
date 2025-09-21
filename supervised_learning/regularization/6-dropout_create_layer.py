#!/usr/bin/env python3
"""This module includes the function that
creates a layer of a neural network using dropout"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Args:
        prev is a tensor containing the output of the previous layer
        n is the number of nodes the new layer should contain
        activation is the activation function for the new layer
        keep_prob is the probability that a node will be kept
        training is a boolean indicating whether the model is in training mode
    Returns:
        the output of the new layer
    """
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode='fan_avg'
    )
    dense_layer = tf.keras.layers.Dense(n,
                                        activation=activation,
                                        kernel_initializer=initializer)(prev)
    # Creating Dropout layer
    if training:
        dropout = tf.nn.dropout(dense_layer, rate=1 - keep_prob)
    return dropout
