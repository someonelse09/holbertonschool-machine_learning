#!/usr/bin/env python3
"""This module includes the class GaussianProcess
 that represents a noiseless 1D Gaussian process"""

import numpy as np


class GaussianProcess:
    """
        Represents a noiseless 1D Gaussian process
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Args:
            X_init is a numpy.ndarray of shape (t, 1) representing the
             inputs already sampled with the black-box function
            Y_init is a numpy.ndarray of shape (t, 1) representing the
             outputs of the black-box function for each input in X_init
            t is the number of initial samples
            l is the length parameter for the kernel
            sigma_f is the standard deviation given to the
             output of the black-box function
            Sets the public instance attributes X, Y, l, and sigma_f
             corresponding to the respective constructor inputs
            Sets the public instance attribute K, representing the current
             covariance kernel matrix for the Gaussian process
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """Calculates the covariance kernel matrix between two matrices
        Args:
            X1 is a numpy.ndarray of shape (m, 1)
            X2 is a numpy.ndarray of shape (n, 1)
            the kernel should use the Radial Basis Function (RBF)
        Returns:
            the covariance kernel matrix as a numpy.ndarray of shape (m, n)
        """
        # Calculate squared Euclidean distance using broadcasting
        # ||x1 - x2||^2 = ||x1||^2 + ||x2||^2 - 2*x1*x2
        sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) +\
                np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)

        # RBF kernel: K(x1, x2) = sigma_f^2 * exp(-0.5 * ||x1 - x2||^2 / l^2)
        return self.sigma_f ** 2 * np.exp(-0.5 * sqdist / self.l ** 2)

    def predict(self, X_s):
        """Predicts the mean and standard deviation
        of points in a Gaussian process
        Args:
            X_s is a numpy.ndarray of shape (s, 1) containing all of the
             points whose mean and standard deviation should be calculated
            s is the number of sample points
        Returns:
            mu, sigma
            mu is a numpy.ndarray of shape (s,) containing
             the mean for each point in X_s, respectively
            sigma is a numpy.ndarray of shape (s,) containing the
             variance for each point in X_s, respectively
        """
        # K_s: covariance between training points and test points (t, s)
        K_s = self.kernel(self.X, X_s)
        # K_ss: covariance between test points themselves (s, s)
        K_ss = self.kernel(X_s, X_s)
        # K_inv: inverse of the covariance matrix of training points
        K_inv = np.linalg.inv(self.K)

        # Posterior mean: mu = K_s^T * K^-1 * Y
        mu = K_s.T.dot(K_inv).dot(self.Y)
        # Flatten to shape (s,)
        mu = mu.reshape(-1)
        # Posterior covariance: sigma = K_ss - K_s^T * K^-1 * K_s
        cov = K_ss - K_s.T.dot(K_inv).dot(K_s)
        # Extract diagonal (variance) and return as standard deviation
        return mu, np.diag(cov)

    def update(self, X_new, Y_new):
        """
        Args:
            X_new is a numpy.ndarray of shape (1,)
             that represents the new sample point
            Y_new is a numpy.ndarray of shape (1,)
             that represents the new sample function value
            Updates the public instance attributes X, Y, and K
        """
        X_new = X_new.reshape(-1, 1)
        Y_new = Y_new.reshape(-1, 1)
        # Appending new points to X and Y
        self.X = np.concatenate((self.X, X_new), axis=0)
        self.Y = np.concatenate((self.Y, Y_new), axis=0)

        self.K = self.kernel(self.X, self.X)
