#!/usr/bin/env python3
"""This module includes the function that creates a batch normalization layer for a neural network in tensorflow"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Args:
        prev is the activated output of the previous layer
        n is the number of nodes in the layer to be created
        activation is the activation function that should be used on the output of the layer
        you should use the tf.keras.layers.Dense layer as the base layer with kernel initializer tf.keras.initializers.VarianceScaling(mode='fan_avg')
        your layer should incorporate two trainable parameters, gamma and beta, initialized as vectors of 1 and 0 respectively
        you should use an epsilon of 1e-7
    Returns:
        a tensor of the activated output for the layer
    """
    # Create dense layer without activation
    dense = tf.keras.layers.Dense(
        n,
        kernel_initializer=tf.keras.initializers.VarianceScaling(mode='fan_avg'),
        use_bias=False  # We'll handle bias through beta parameter
    )(prev)

    # Manual batch normalization
    # Calculate batch mean and variance
    mean, variance = tf.nn.moments(dense, axes=[0])

    # Create trainable parameters gamma and beta
    gamma = tf.Variable(tf.ones([n]), trainable=True, name='gamma')
    beta = tf.Variable(tf.zeros([n]), trainable=True, name='beta')

    # Apply batch normalization formula
    epsilon = 1e-7
    normalized = (dense - mean) / tf.sqrt(variance + epsilon)
    batch_normed = gamma * normalized + beta

    # Apply activation
    output = tf.keras.layers.Activation(activation)(batch_normed)

    return output
