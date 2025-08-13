#!/usr/bin/env python3
""" This module includes function called def summation_i_squared(n) """


def summation_i_squared(n):
    """ This function calculates the sum
    of squares until the valid number n """

    if n < 0:
        return None
    elif n == 0:
        return 0
    else:
        return n**2 + summation_i_squared(n - 1)
