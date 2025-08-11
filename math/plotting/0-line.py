#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def line():

    x = np.arange(0, 11)
    y = x ** 3
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(x, y, 'r-')
    plt.show()
