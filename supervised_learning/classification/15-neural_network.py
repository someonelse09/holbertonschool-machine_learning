#!/usr/bin/env python3
"""This module includes the class Neural Network"""

import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """This class defines a Neural Network"""
    def __init__(self, nx, nodes):
        """Initialising the Neural Network"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
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
        return self.__W2

    @property
    def b2(self):
        """getter method for output layer's bias"""
        return self.__b2

    @property
    def A2(self):
        """getter method for
        output layer's activated output"""
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation
         of the neural network"""
        # Z1 = W1 * X + b1
        Z1 = np.matmul(self.__W1, X) + self.__b1
        A1 = 1/(1 + np.exp(-Z1))
        # Z2 = W2 * A1 + b2
        Z2 = np.matmul(self.__W2, A1) + self.__b2
        A2 = 1/(1 + np.exp(-Z2))
        self.__A1 = A1
        self.__A2 = A2
        return A1, A2

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        cost = -((1/m)*np.sum(Y*np.log(A) + (1 - Y)*np.log(1.0000001 - A)))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        A1, A2 = self.forward_prop(X)
        prediction = np.where(A2 >= 0.5, 1, 0)
        cost = self.cost(Y, A2)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        m = Y.shape[1]
        dZ2 = A2 - Y
        dW2 = (1/m)*np.matmul(dZ2, A1.T)
        db2 = (1/m)*np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.matmul(self.__W2.T, dZ2)*A1*(1 - A1)
        dW1 = (1/m)*np.matmul(dZ1, X.T)
        db1 = (1/m)*np.sum(dZ1, axis=1, keepdims=True)

        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2
        self.__W1 -= alpha*dW1
        self.__b1 -= alpha*db1

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Method to train the neural network"""
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

        for iteration in range(iterations + 1):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)
            if iteration % step == 0 or iteration == iterations:
                cost = self.cost(Y, A2)
                costs.append(cost)
                iters.append(iteration)
                if verbose:
                    print(f"Cost after {iteration} iterations: {cost}")
        if graph:
            plt.plot(iters, costs, 'b-')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
        return self.evaluate(X, Y)
