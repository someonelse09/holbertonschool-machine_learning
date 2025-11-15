#!/usr/bin/env python3
"""This module includes the class Binomial
that represents a binomial distribution"""


def factorial(n):
    """Calculating Factorial"""
    res = 1
    for i in range(2, n + 1):
        res *= i
    return res


def combination(n, k):
    """Calculating combination of n taken by k"""
    return factorial(n) // (factorial(k) * factorial(n - k))


class Binomial:
    """Binomial Distribution"""
    def __init__(self,data=None, n=1, p=0.5):
        """
        Args:
            ata is a list of the data to be used to estimate the distribution
            n is the number of Bernoulli trials
            p is the probability of a “success”
            Sets the instance attributes n and p
            Saves n as an integer and p as a float
            If data is not given (i.e. None)
            Use the given n and p
            If n is not a positive value, raise a ValueError
             with the message n must be a positive value
            If p is not a valid probability, raise a ValueError with
             the message p must be greater than 0 and less than 1
            If data is given:
            Calculate n and p from data
            Round n to the nearest integer
             (rounded, not casting! int(3.7)
              is not the same as round(3.7))
            Hint: Calculate p first and then calculate n.
             Then recalculate p. Think about
              why you would want to do it this way?
            If data is not a list, raise a TypeError
             with the message data must be a list
            If data does not contain at least
             two data points, raise a ValueError with the
              message data must contain multiple values
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            var = sum((x - mean) ** 2 for x in data) / len(data)
            self.p = 1 - (var / mean)
            self.n = round(mean / self.p)
            self.p = mean / self.n

            self.n = int(self.n)
            self.p = float(self.p)

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of “successes”
        Args:
            k is the number of “successes”
            If k is not an integer, convert it to an integer
            If k is out of range, return 0
        Returns:
            the PMF value for k
        """
        if not isinstance(k, int):
            k = int(k)
        c = combination(self.n, k)
        pmf = c * (self.p ** k) * ((1 - self.p) ** (self.n - k))
        return pmf

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of “successes”
        Args:
            k is the number of “successes”
            If k is not an integer, convert it to an integer
            If k is out of range, return 0
        Returns:
            the CDF value for k
        """
        if not isinstance(k, int):
            k = int(k)
        cdf = 0
        for i in range(1, k + 1):
            cdf += self.pmf(i)
        return cdf
