#!/usr/bin/env python3
""" This module contains function called gradient """

import numpy as np
import matplotlib.pyplot as plt


def gradient():
    """ This function draws a
    scatter plot of mountain elevation """
    np.random.seed(5)

    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))
    plt.figure(figsize=(6.4, 4.8))

    sct = plt.scatter(x, y, c=z, cmap='viridis')
    plt.xlabel("x coordinate (m)")
    plt.ylabel("y coordinate (m)")
    plt.title("Mountain Elevation")
    cbar = plt.colorbar(sct)
    cbar.set_label("elevation (m)")
    plt.show()
