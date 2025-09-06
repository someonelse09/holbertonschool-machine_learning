#!/usr/bin/env python3
"""This module includes the function
for building a model with Tensorflow"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
import numpy as np


def build_model(nx, layers, activations, lambtha, keep_prob):
    """This function builds a neural network
    using Input and Model classes of Keras"""
    inputs = Input(shape=(nx,))
    m = Dense(units=layers[0],
                    activations=activations[0],
                    kernel_regularizer=l2(lambtha))(inputs)
    if keep_prob < 1:
        m = Dropout(1 - keep_prob)(m)
    for i in range(1, len(layers)):
        m = (units=layers[i],
                  activations=activations[i],
                  kernel_regularizer=l2(lambtha))(m)
    if i < len(layers) and keep_prob < 1:
        m = (Dropout(1 - keep_prob))(m)
    model = Model(inputs=inputs, outputs=m)
    return model
