#!/usr/bin/env python3
"""This module includes the function that
builds an identity block as described in
Deep Residual Learning for Image Recognition (2015)"""

from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Args:
        A_prev is the output from the previous layer
        filters is a tuple or list containing F11, F3, F12, respectively:
        F11 is the number of filters in the first 1x1 convolution
        F3 is the number of filters in the 3x3 convolution
        F12 is the number of filters in the second 1x1 convolution
        All convolutions inside the block should be followed by
         batch normalization along the channels axis and
          a rectified linear activation (ReLU), respectively.
        All weights should use he normal initialization
        The seed for the he_normal initializer should be set to zero
    Returns:
        the activated output of the identity block
    """
    F11, F3, F12 = filters
    initializers = K.initializers.HeNormal(seed=0)
    temp = A_prev

    X = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initializers
    )(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initializers
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initializers
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    X = K.layers.Add()([X, temp])
    X = K.layers.Activation('relu')(X)

    return X
