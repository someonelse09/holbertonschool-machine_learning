#!/usr/bin/env python3
"""This module includes the class DeepNeuralNetwork"""

import numpy as np


class DeepNeuralNetwork:
    """This class is for implementing
     multi-layered(more than two) Neural Networks"""
    def __init__(self, nx, layers):
        """Initalising the Deep Neural Network"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for lx in range(1, self.L + 1):
            if not isinstance(layers[lx - 1], int) and layers[lx - 1] > 0:
                raise ValueError("layers must be a list of positive integers")
            if lx == 1:
                previous_nodes = nx
            else:
                previous_nodes = layers[lx - 2]
            self.weights['W' + str(lx)] = (
                np.random.randn(layers[lx - 1], previous_nodes)
                * np.sqrt(2/previous_nodes)
            )
            self.weights['b' + str(lx)] = np.zeros((layers[lx - 1], 1))
