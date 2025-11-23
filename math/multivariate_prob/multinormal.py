#!/usr/bin/env python3
"""This module includes the class
that represents a Multivariate Normal distribution"""

import numpy as np


class MultiNormal:
    """
    Set the public instance variables:
        mean - a numpy.ndarray of shape
         (d, 1) containing the mean of data
        cov - a numpy.ndarray of shape
         (d, d) containing the covariance matrix data
    """
    def __init__(self, data):
        """
        Args:
            data is a numpy.ndarray of shape
             (d, n) containing the data set:
            n is the number of data points
            d is the number of dimensions in each data point
        If data is not a 2D numpy.ndarray, raise a TypeError
         with the message data must be a 2D numpy.ndarray
        If n is less than 2, raise a ValueError with the
         message data must contain multiple data points
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a 2D numpy.ndarray")

            # Check if data is 2D
        if len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)

        data_centered = data - self.mean

        self.cov = (data_centered @ data_centered.T) / (n - 1)

    def pdf(self, x):
        """
        Calculates the PDF at a data point

        Args:
            x: numpy.ndarray of shape (d, 1) containing the data point

        Returns:
            The value of the PDF at x
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]

        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))


        # Calculate determinant of covariance matrix
        det_cov = np.linalg.det(self.cov)

        # Calculate inverse of covariance matrix
        inv_cov = np.linalg.inv(self.cov)

        # Calculate (x - mean)
        x_centered = x - self.mean

        # Calculate the exponent: -0.5 * (x-μ)^T * Σ^(-1) * (x-μ)
        exponent = -0.5 * (x_centered.T @ inv_cov @ x_centered)

        # Calculate the coefficient: 1 / sqrt((2π)^d * |Σ|)
        coefficient = 1.0 / np.sqrt(((2 * np.pi) ** d) * det_cov)

        pdf_value = coefficient * np.exp(exponent)

        return float(pdf_value)
