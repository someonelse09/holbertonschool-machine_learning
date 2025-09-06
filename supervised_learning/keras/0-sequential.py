#!/usr/bin/env python3
"""This module includes function
for building a model"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
import numpy as np


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Building a Neural Network with
    Sequential model fromKeras API"""
    model = Sequential()
    model.add(Dense(units=layers[0],
                    activations=activations[0],
                    kernel_regularizer=l2(lambtha),
                    input_shape=(nx,)))
    if keep_prob < 1:
        model.add(Dropout(1 - keep_prob))
    for i in range(1, len(layers)):
        model.add(Dense(units=layers[i],
                  activations=activations[i],
                  kernel_regularizer=l2(lambtha)))
        if i < len(layers) - 1 and keep_prob < 1:
            model.add(Dropout(1 - keep_prob))
    return model
