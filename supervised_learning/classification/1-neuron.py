#!/usr/bin/env python3
"""This module includes the Neuron class"""

import numpy as np


class Neuron:
    """This class defines a general
    structure for a neuron"""
    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """getter method for weights"""
        return self.__W

    @property
    def b(self):
        """getter method for variable bias"""
        return self.__b

    @property
    def A(self):
        """getter method for
        The activated output of the neuron (prediction)"""
        return self.__A
