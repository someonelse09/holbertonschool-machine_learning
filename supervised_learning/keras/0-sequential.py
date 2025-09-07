#!/usr/bin/env python3
"""This module includes the function
which generates models using tensorflow keras"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Building a Neural Network with
    Sequential model fromKeras API"""
    model = K.Sequential()
    model.add(K.layers.Dense(units=layers[0],
              activation=activations[0],
              kernel_regularizer=K.regularizers.l2(lambtha),
              input_shape=(nx,)))
    if keep_prob < 1 and len(layers) > 1:
        model.add(K.layers.Dropout(1 - keep_prob))
    for i in range(1, len(layers)):
        model.add(K.layers.Dense(units=layers[i],
                  activation=activations[i],
                  kernel_regularizer=K.regularizers.l2(lambtha)))
        if i < len(layers) - 1 and keep_prob < 1:
            model.add(K.layers.Dropout(1 - keep_prob))
    return model
