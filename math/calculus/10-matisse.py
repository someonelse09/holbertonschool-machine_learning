#!/usr/bin/env python3
""" This module contain the function called matisse """


def poly_derivative(poly):
    """ This function calculates the
    derivative of the given polynomial """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not all(isinstance(c, (int, float)) for c in poly):
        return None
    elif len(poly) == 1:
        return [0]
    return [x*poly[x] for x in range(1, len(poly))]
