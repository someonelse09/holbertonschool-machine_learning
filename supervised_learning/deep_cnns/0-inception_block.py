#!/usr/bin/env python3
"""This module includes the function that builds an
inception block as described in Going Deeper with Convolutions (2014)"""

from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    Args:
        A_prev is the output from the previous layer
        filters is a tuple or list containing
         F1, F3R, F3,F5R, F5, FPP, respectively:
        F1 is the number of filters in the 1x1 convolution
        F3R is the number of filters in the
         1x1 convolution before the 3x3 convolution
        F3 is the number of filters in the 3x3 convolution
        F5R is the number of filters in the
         1x1 convolution before the 5x5 convolution
        F5 is the number of filters in the 5x5 convolution
        FPP is the number of filters in the
         1x1 convolution after the max pooling
        All convolutions inside the inception block
         should use a rectified linear activation (ReLU)
    Returns:
        the concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    # First Path: 1x1 convolution
    conv1x1 = K.layers.Conv2D(
        filters=F1,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(A_prev)

    # Second Path: 1x1 convolution which is followed by 3x3 convolution
    conv3x3_reduce = K.layers.Conv2D(
        filters=F3R,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(A_prev)

    conv3x3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    )(conv3x3_reduce)

    # Third Path: 1x1 convolution which is followed by 5x5 convolution
    conv5x5_reduce = K.layers.Conv2D(
        filters=F5R,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(A_prev)

    conv5x5 = K.layers.Conv2D(
        filters=F5,
        kernel_size=(5, 5),
        padding='same',
        activation='relu'
    )(conv5x5_reduce)

    # Fourth Path: Max Pooling that is followed by 1x1 convolution
    max_pool = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding='same'
    )(A_prev)

    conv1x1_pool = K.layers.Conv2D(
        filters=FPP,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(max_pool)

    filter_concatenation = K.layers.concatenate([conv1x1, conv3x3, conv5x5, conv1x1_pool])

    return filter_concatenation
