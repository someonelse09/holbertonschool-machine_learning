#!/usr/bin/env python3
""" This Module includes the
function named line """


import numpy as np
import matplotlib.pyplot as plt


def line():
    """ This function draws the
    line graph determined by the code below """

    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    x = np.arange(0, 11)
    plt.plot(x, y, 'r-')
    plt.axis((0, 10, None, None))
    plt.show()
