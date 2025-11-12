#!/usr/bin/env python3
"""This module includes the class Poisson
that represents a poisson distribution"""

import numpy as np
import math

class Poisson:
    def __init__(self, data=None, lambtha=1.):
        """
        Args:
            data is a list of the data to be
             used to estimate the distribution
            lambtha is the expected number of
             occurences in a given time frame
            Sets the instance attribute lambtha
            Saves lambtha as a float
            If data is not given, (i.e. None (be careful:
             not data has not the same result as data is None)):
            Use the given lambtha
            If lambtha is not a positive value or equals to 0,
             raise a ValueError with the message
              lambtha must be a positive value
            If data is given:
            Calculate the lambtha of data
            If data is not a list, raise a TypeError
             with the message data must be a list
            If data does not contain at least two data points,
             raise a ValueError with the message data
              must contain multiple values
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
            self.data = data
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculates the value of the PMF for
         a given number of “successes”
        Args:
            k is the number of “successes”
            If k is not an integer, convert it to an integer
            If k is out of range, return 0
        Returns:
            the PMF value for k
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        pmf = (np.exp(-self.lambtha) *
               self.lambtha ** k) / math.factorial(k)
        return pmf

    def cdf(self, k):
        """
        Calculates the value of the CDF
         for a given number of “successes”
        Args:
            k is the number of “successes”
            If k is not an integer, convert it to an integer
            If k is out of range, return 0
        Returns:
            the CDF value for k
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
