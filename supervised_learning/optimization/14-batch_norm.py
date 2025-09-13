#!/usr/bin/env python3
"""This module includes the function
that creates a batch normalization layer
for a neural network in tensorflow"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Args:
        prev is the activated output of the previous layer
        n is the number of nodes in the layer to be created
        activation is the activation function
            that should be used on the output of the layer
        you should use the tf.keras.layers.Dense layer
            as the base layer with kernel initializer
                tf.keras.initializers.VarianceScaling(mode='fan_avg')
        your layer should incorporate two trainable parameters,
            gamma and beta, initialized as vectors of 1 and 0 respectively
        you should use an epsilon of 1e-7
    Returns:
        a tensor of the activated output for the layer
    """
    dense = tf.keras.layers.Dense(
        n,
        kernel_initializer=tf.keras.initializers.VarianceScaling(mode='fan_avg')
    )(prev)
    batch_norm = tf.keras.layers.BatchNormalization(
        axis=-1,
        epsilon=1e-7,
        scale=True,
        beta_initializer='zeros',
        gamma_initializer='ones'
    )(dense)

    output = tf.keras.layers.Activation(activation)(batch_norm)

    return output
