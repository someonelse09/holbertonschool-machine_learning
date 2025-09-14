#!/usr/bin/env python3
"""This module includes the
function called moving_average"""


def moving_average(data, beta):
    """Function that calculates the
    weighted moving average of a data set"""
    v = 0
    moving_averages = []
    for index, item in enumerate(data, 1):
        v = beta * v + (1 - beta) * item
        v_adjusted = v / (1 - beta ** index)
        moving_averages.append(v_adjusted)
    return moving_averages

# for i in range(len(data)):
#     x = data[i]
#     t = i + 1   # because bias correction uses 1-based indexing
#     v = beta * v + (1 - beta) * x
#     v_corrected = v / (1 - beta ** t)
#     moving_averages.append(v_corrected)
