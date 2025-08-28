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
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
