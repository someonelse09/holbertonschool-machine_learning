#!/usr/bin/env python3
"""This module contains the class Normal
that represents a normal distribution"""

pi = 3.1415926535
e = 2.7182818285


class Normal:
    """Gaussian Distribution"""
    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Args:
            data is a list of the data to be used to estimate the distribution
            mean is the mean of the distribution
            stddev is the standard deviation of the distribution
            Sets the instance attributes mean and stddev
            Saves mean and stddev as floats
            If data is not given (i.e. None (be careful:
             not data has not the same result as data is None))
            Use the given mean and stddev
            If stddev is not a positive value or equals to 0,
             raise a ValueError with the message
              stddev must be a positive value
            If data is given:
            Calculate the mean and standard deviation of data
            If data is not a list, raise a
             TypeError with the message data must be a list
            If data does not contain at least two data points,
             raise a ValueError with the message
              data must contain multiple values
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.stddev = float(stddev)
            self.mean = float(mean)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            mse = 0
            for r in data:
                mse += (r - self.mean) ** 2
            self.stddev = (mse / len(data)) ** 0.5

    # z = (x - mean) / standard deviation
    def z_score(self, x):
        """Calculates the z-score of a given x-value
        Args:
            x is the x-value
        Returns:
            the z-score of x
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score
        Args:
            z is the z-score
        Returns:
            the x-value of z
        """
        return z * self.stddev + self.mean

    def pdf(self, x):
        """Calculates the value of the PDF for a given x-value
        Args:
            x is the x-value
        Returns:
            the PDF value for x
        """
        c = self.stddev * ((2 * pi) ** 0.5)
        exponent = -(1 / 2) * ((x - self.mean) / self.stddev) ** 2
        pdf = float((1 / c) * e ** exponent)
        return pdf

    def erf(self, x):
        """Approximates the error function"""
        pi = 3.1415926535
        coef = 2 / (pi ** 0.5)
        first = (x ** 3) / 3
        second = (x ** 5) / 10
        third = (x ** 7) / 42
        fourth = (x ** 9) / 216
        return coef * (x - first + second - third + fourth)

    def cdf(self, x):
        """Calculates the CDF for a given x-value"""
        z = (x - self.mean) / (self.stddev * (2 ** 0.5))
        return 0.5 * (1 + self.erf(z))
