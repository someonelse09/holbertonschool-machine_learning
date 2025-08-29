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

    def forward_prop(self, X):
        """This method Calculates the
        forward propagation of the neuron"""
        Z = np.matmul(self.__W, X) + self.__b
        A = 1/(1 + np.exp(-Z))
        self.__A = A
        return A

    def cost(self, Y, A):
        """Calculates the cost of the
        model using logistic regression"""
        m = Y.shape[1]
        Cost = -((1/m)*(np.sum((1 - Y)*np.log(1.0000001 - A) + Y*np.log(A))))
        return Cost

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""
        A = self.forward_prop(X)
        predictions = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return predictions, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""
        m = X.shape[1]
        dZ = A - Y
        dW = (1/m)*np.dot(dZ, X.T)
        db = (1/m)*np.sum(dZ)
        self.__W = self.__W - alpha*dW
        self.__b = self.__b - alpha*db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """This module trains the neuron"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
        return self.evaluate(X, Y)
