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
        """Forward propagation for multiclass classification"""
        self.__cache['A0'] = X
        for lx in range(1, self.__L + 1):
            W = self.__weights['W' + str(lx)]
            b = self.__weights['b' + str(lx)]
            A_prev = self.__cache['A' + str(lx - 1)]

            Z = np.matmul(W, A_prev) + b
            if lx == self.__L:
                # Softmax for last layer (multiclass output)
                exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            else:
                # Sigmoid for hidden layers
                A = 1 / (1 + np.exp(-Z))

            self.__cache['A' + str(lx)] = A
        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculates the cost using categorical cross-entropy for multiclass"""
        m = Y.shape[1]
        # Categorical cross-entropy loss
        # Clip A to prevent log(0) while maintaining numerical stability
        A_clipped = np.clip(A, 1e-15, 1 - 1e-15)
        cost = -np.sum(Y * np.log(A_clipped)) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions for multiclass"""
        A, _ = self.forward_prop(X)
        # Convert softmax probabilities to one-hot predictions
        predictions = np.zeros_like(A)
        max_indices = np.argmax(A, axis=0)
        predictions[max_indices, np.arange(A.shape[1])] = 1
        
        cost = self.cost(Y, A)
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
