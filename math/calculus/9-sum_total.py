#!/usr/bin/env python3
""" This module includes function called def summation_i_squared(n) """


def summation_i_squared(n):
    """ This function calculates the sum
    of squares until the valid number n """

    if n < 0:
        return None
    return sum(i**2 for i in range(n + 1))
