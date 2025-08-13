#!/usr/bin/env python3
""" This module includes function called def summation_i_squared(n) """


def summation_i_squared(n):
    """ This function calculates the sum
    of squares until the valid number n """

    if n < 1:
        return None
    return n*(n + 1)*(2*n + 1)/6
