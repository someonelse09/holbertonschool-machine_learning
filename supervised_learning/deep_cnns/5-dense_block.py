#!/usr/bin/env python3
"""This module includes the function that
builds a dense block as described in
Densely Connected Convolutional Networks"""

from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Args:
        X is the output from the previous layer
        nb_filters is an integer representing the number of filters in X
        growth_rate is the growth rate for the dense block
        layers is the number of layers in the dense block
        You should use the bottleneck layers used for DenseNet-B
        All weights should use he normal initialization
        The seed for the he_normal initializer should be set to zero
        All convolutions should be preceded by
         Batch Normalization and
          a rectified linear activation (ReLU), respectively
    Returns:
        The concatenated output of each layer within the
         Dense Block and the number of filters
          within the concatenated outputs, respectively
    """
    initializer = K.initializers.HeNormal(seed=0)

    concatenated = X

    for i in range(layers):
        # Bottleneck layer: BN -> ReLU -> Conv 1x1
        # 1x1 conv produces 4 * growth_rate feature maps
        bn1 = K.layers.BatchNormalization(axis=3)(concatenated)
        act1 = K.layers.Activation('relu')(bn1)
        conv1x1 = K.layers.Conv2D(
            filters=4 * growth_rate,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            kernel_initializer=initializer
        )(act1)

        # Composite layer: BN -> ReLU -> Conv 3x3
        # 3x3 conv produces growth_rate feature maps
        bn2 = K.layers.BatchNormalization(axis=3)(conv1x1)
        act2 = K.layers.Activation('relu')(bn2)
        conv3x3 = K.layers.Conv2D(
            filters=growth_rate,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            kernel_initializer=initializer
        )(act2)

        concatenated = K.layers.concatenate([concatenated, conv3x3])

        nb_filters += growth_rate

    return concatenated, nb_filters
