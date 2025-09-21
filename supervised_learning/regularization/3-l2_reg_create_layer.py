#!/usr/bin/env python3
"""This module includes function that
creates a neural network layer
in tensorFlow that includes L2 regularization"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Args:
        prev is a tensor containing the output of the previous layer
        n is the number of nodes the new layer should contain
        activation is the activation function that should be used on the layer
        lambtha is the L2 regularization parameter
    Returns:
        the output of the new layer"""
    layer_weight = tf.initializers.VarianceScaling(
        scale=2.0, mode='fan_avg'
    )
    regularizer = tf.keras.regularizers.L2(lambtha)
    layer = tf.keras.layers.Dense(n,
                                  activation=activation,
                                  kernel_initializer=layer_weight,
                                  kernel_regularizer=regularizer)

    return layer(prev)
