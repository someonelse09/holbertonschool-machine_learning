#!/usr/bin/env python3
"""This module includes the class DeepNeuralNetwork"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle


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
            A = 1 / (1 + np.exp(-Z))
            self.__cache['A' + str(lx)] = A
        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculates the cost of
        the model using logistic regression"""
        m = Y.shape[1]
        cost = -((1/m)*(
            np.sum((1 - Y) * np.log(1.0000001 - A) + Y * np.log(A))
        ))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        Al, _ = self.forward_prop(X)
        predictions = np.where(Al >= 0.5, 1, 0)
        cost = self.cost(Y, Al)
        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculate one pass of gradient descent on the neural network."""
        m = Y.shape[1]
        AL = cache['A{}'.format(self.__L)]
        dZl = AL - Y
        for i in range(self.__L, 0, -1):
            Al = cache['A{}'.format(i-1)]
            dwl = (dZl @ Al.T) / m
            dbl = (np.sum(dZl, axis=1, keepdims=True)) / m

            Al_prev = cache['A{}'.format(i-1)]
            Wl = self.__weights['W{}'.format(i)]
            if i > 1:
                dZl = (Wl.T @ dZl) * (Al_prev * (1-Al_prev))
            self.__weights['W{}'.format(i)] -= alpha * dwl
            self.__weights['b{}'.format(i)] -= alpha * dbl

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the deep neural network"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        iters = []

        for i in range(iterations + 1):
            Al, cache = self.forward_prop(X)
            if i % step == 0 or i == iterations:
                cost = self.cost(Y, Al)
                costs.append(cost)
                iters.append(i)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))
            # skip gradient update after final forward pass
            if i < iterations:
                self.gradient_descent(Y, cache, alpha)
        if graph:
            plt.plot(iters, costs, 'b-')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format"""
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        return filename

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object"""
        if not os.path.exists(filename):
            return None
        with open(filename, "rb") as f:
            return pickle.load(f)
