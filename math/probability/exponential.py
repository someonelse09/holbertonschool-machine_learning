#!/usr/bin/env python3
"""This module includes the class Exponential
that represents an exponential distribution"""

e = 2.7182818285


class Exponential:
    """Exponential Distribution"""
    def __init__(self, data=None, lambtha=1.):
        """
        Args:
            data is a list of the data to be
             used to estimate the distribution
            lambtha is the expected number of
             occurences in a given time frame
            Sets the instance attribute lambtha
            Saves lambtha as a float
            If data is not given (i.e. None):
            Use the given lambtha
            If lambtha is not a positive value,
             raise a ValueError with the message
              lambtha must be a positive value
            If data is given:
            Calculate the lambtha of data
            If data is not a list, raise a TypeError
             with the message data must be a list
            If data does not contain at least two data points,
             raise a ValueError with the message
              data must contain multiple values
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(len(data) / sum(data))
            self.data = data

    def pdf(self, x):
        """Calculates the value of the PDF for a given time period
        Args:
            x is the time period
            If x is out of range, return 0
        Returns:
            the PDF value for x
        """
        if x < 0:
            return 0
        pdf = self.lambtha * e ** (-self.lambtha * x)
        return pdf

    def cdf(self, x):
        """Calculates the value of the CDF for a given time period
        Args:
            x is the time period
            If x is out of range, return 0
        Returns:
            the CDF value for x
        """
        if x < 0:
            return 0
        cdf = 1.0 - e ** (-self.lambtha * x)
        return cdf
