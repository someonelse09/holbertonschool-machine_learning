#!/usr/bin/env python3
"""This module includes the
function called normalize"""

import numpy as np


def normalize(X, m, s):
    """This function returns the normalized tensor X"""
    return (X - m) / s
