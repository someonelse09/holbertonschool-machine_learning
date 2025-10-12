#!/usr/bin/env python3
"""This module includes function that
builds a projection block as described in
Deep Residual Learning for Image Recognition (2015)"""

from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Args:
        A_prev is the output from the previous layer
        filters is a tuple or list containing F11, F3, F12, respectively:
        F11 is the number of filters in the first 1x1 convolution
        F3 is the number of filters in the 3x3 convolution
        F12 is the number of filters in the second 1x1 convolution
         as well as the 1x1 convolution in the shortcut connection
        s is the stride of the first convolution in
         both the main path and the shortcut connection
        All convolutions inside the block should be followed by
         batch normalization along the channels axis and
          a rectified linear activation (ReLU), respectively.
        All weights should use he normal initialization
        The seed for the he_normal initializer should be set to zero
    Returns:
        the activated output of the projection block
    """
    F11, F3, F12 = filters
    initializer = K.initializers.HeNormal(seed=0)

    # Main Path 1x1 convolution
    X = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=s,
        padding='same',
        kernel_initializer=initializer
    )(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Second component of main path
    X = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Third component of main path
    X = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # Shortcut Path
    shortcut = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=s,
        padding='same',
        kernel_initializer=initializer
    )(A_prev)
    shortcut = K.layers.BatchNormalization(axis=3)(shortcut)

    X = K.layers.Add()([X, shortcut])

    X = K.layers.Activation('relu')(X)

    return X
