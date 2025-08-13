#!/usr/bin/env python3
""" This module contain the function called poly_integral """


def poly_integral(poly, C=0):
    """ This function calculates the
    integral of the given polynomial """

    if not isinstance(poly, list):
        return None
    if not all(isinstance(s, (int, float)) for s in poly):
        return None
    if not isinstance(C, (int, float)):
        return None
    elif len(poly) == 0:
        return None
    result = [C]
    for x in range(len(poly)):
        val = poly[x]/(x + 1)
        if val.is_integer():
            val = int(val)
        result.append(val)
    return result
