#!/usr/bin/env python3
"""Builds a neural network with the Keras library"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network with the Keras Functional API"""
    inputs = K.Input(shape=(nx,))
    x = K.layers.Dense(
        units=layers[0],
        activation=activations[0],
        kernel_regularizer=K.regularizers.l2(lambtha)
    )(inputs)

    if keep_prob < 1 and  len(layers) > 1:
        x = K.layers.Dropout(1 - keep_prob)(x)

    for i in range(1, len(layers)):
        x = K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )(x)

        if i < len(layers) - 1 and keep_prob < 1:
            x = K.layers.Dropout(1 - keep_prob)(x)

    model = K.Model(inputs=inputs, outputs=x)
    return model
