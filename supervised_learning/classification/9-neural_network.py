#!/usr/bin/env python3
"""This module includes the class Neural Network"""

import numpy as np


class NeuralNetwork:
    """This class defines a Neural Network"""
    def __init__(self, nx, nodes):
        """Initialising the Neural Network"""
        if not isinstance(nx, int):
            raise TypeError("x must be an integer")
        if nx <= 0:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes <= 0:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """getter method for hidden layer's weight"""
        return self.__W1

    @property
    def b1(self):
        """getter method for hidden layer's bias"""
        return self.__b1

    @property
    def A1(self):
        """getter method for
        hidden layer's activated output"""
        return self.__A1

    @property
    def W2(self):
        """getter method for output layer's weight"""
        return self.__W1

    @property
    def b2(self):
        """getter method for output layer's bias"""
        return self.__b1

    @property
    def A2(self):
        """getter method for
        output layer's activated output"""
        return self.__A2
