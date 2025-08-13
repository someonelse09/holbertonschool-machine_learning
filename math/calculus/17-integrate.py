#!/usr/bin/env python3
""" This module contain the function called poly_integral """


def poly_integral(poly, C=0):
    """ This function calculates the
    integral of the given polynomial """

    if not isinstance(poly, list):
        return None
    if not all(isinstance(c, (int, float)) for c in poly):
        return None
    elif len(poly) == 0:
        return [0]
    result = [1/(x)*poly[x - 1] for x in range(1, len(poly) + 1)]
    result.insert(0, 0)
    return result
