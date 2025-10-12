#!/usr/bin/env python3
"""This module includes the function that
builds the DenseNet-121 architecture
as described in Densely Connected Convolutional Networks"""

from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer

def densenet121(growth_rate=32, compression=1.0):
    """
    Args:
    growth_rate is the growth rate
        compression is the compression factor
        You can assume the input data will have shape (224, 224, 3)
        All convolutions should be preceded by Batch Normalization
         and a rectified linear activation (ReLU), respectively
        All weights should use he normal initialization
        The seed for the he_normal initializer should be set to zero
        You may use:
        dense_block = __import__('5-dense_block').dense_block
        transition_layer = __import__('6-transition_layer').transition_layer
    Returns:
        the keras model
    """
    initializer = K.initializers.HeNormal(seed=0)
    input_layer = K.Input(shape=(224, 224, 3))

    # Initial layers: BN -> ReLU -> Conv -> MaxPool
    X = K.layers.BatchNormalization(axis=3)(input_layer)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same',
        kernel_initializer=initializer
    )(X)

    pool = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same',
    )(X)
    nb_filters = 64

    X, nb_filters = dense_block(X, nb_filters, growth_rate, 6)

    X, nb_filters = transition_layer(X, nb_filters, compression)

    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)

    X, nb_filters = transition_layer(X, nb_filters, compression)

    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)

    X, nb_filters = transition_layer(X, nb_filters, compression)

    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)

    X = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=(1, 1),
        padding='valid'
    )(X)

    X = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=initializer
    )(X)

    model = K.models.Model(inputs=input_layer, outputs=X)

    return model
