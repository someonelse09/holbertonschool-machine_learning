#!/usr/bin/env python3
"""This module includes the class DeepNeuralNetwork"""

import numpy as np


class DeepNeuralNetwork:
    """This class is for implementing
     multi-layered(more than two) Neural Networks"""

    def __init__(self, nx, layers):
        """Initialising the Deep Neural Network"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for lx in range(1, self.__L + 1):
            # Validate each layer element during the loop
            if not isinstance(layers[lx - 1], int) or layers[lx - 1] <= 0:
                raise TypeError("layers must be a list of positive integers")

            if lx == 1:
                previous_nodes = nx
            else:
                previous_nodes = layers[lx - 2]

            self.weights['W' + str(lx)] = (
                np.random.randn(layers[lx - 1], previous_nodes) *
                np.sqrt(2 / previous_nodes)
            )
            self.weights['b' + str(lx)] = np.zeros((layers[lx - 1], 1))

    @property
    def L(self):
        """getter for number of layers in the DNN"""
        return self.__L

    @property
    def cache(self):
        """getter method for
        intermediary values of the DNN"""
        return self.__cache

    @property
    def weights(self):
        """getter method for weights
        and biases of the DNN"""
        return self.__weights

    def forward_prop(self, X):
        """This method calculates the forward
         propagation of the neural network"""
        self.__cache['A0'] = X
        for lx in range(1, self.__L + 1):
            W = self.__weights['W' + str(lx)]
            b = self.__weights['b' + str(lx)]
            previous_A = self.__cache['A' + str(lx - 1)]

            Z = np.matmul(W, previous_A) + b
            A = 1/(1 + np.exp(-Z))
            self.__cache['A' + str(lx)] = A
        return self.__cache['A' + str(self.__L)], self.__cache
